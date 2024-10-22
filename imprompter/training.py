from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from dataclasses import dataclass
from tqdm import tqdm
import json
import os
from os.path import join
import utils as utils
import urllib3
import pickle
import random
import heapq
import time
import pickle
import numpy as np

IGNORE_INDEX = -100


@dataclass
class FullPrompt:
    """
    Holds a single user prompt, documents, suffix
    """

    prompt_ids: list #torch.Tensor
    suffix_slice: list #slice
    # The targets are the docs
    target_prefix_slice: list #slice
    target_prefix_ids: list #torch.Tensor
    prompt_ident: int  # For bookkeeping purposes

    def update_suffix(self, suffix_ids: torch.Tensor) -> None:
        """
        Updates the prompt with a new suffix
        """
        for i in range(len(self.prompt_ids)):
            self.prompt_ids[i] = torch.cat(
                [
                    self.prompt_ids[i][:, : self.suffix_slice[i].start],
                    suffix_ids.unsqueeze(0).to(self.prompt_ids[i].device),
                    self.prompt_ids[i][:, self.suffix_slice[i].stop :], 
                ],
                dim = -1
            )


class Imprompter(object):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        batch_size: int,
        num_epochs: int,
        k: int,
        n_proposals: int,
        subset_size: int,
        initial_suffix: str,
        vocab: str = "",
        outfile_prefix: str = "niubi",
        start_from_file: str = '',
        reuse_log: str = False,
        autorestart: int = 50,
        syntax_weight: float = 0,
        mask_unrecognizable_unicode = True,
        temperature: float = 0.0,
        ppl_preference: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_epochs = num_epochs
        self.k = k
        self.num_proposals = n_proposals
        self.subset_size = subset_size
        self.vocab = vocab
        self.outfile_prefix = outfile_prefix
        self.autorestart = autorestart
        self.syntax_weight = syntax_weight
        self.temperature = temperature
        self.ppl_preference = ppl_preference

        assert initial_suffix, "initial_suffix must be a non empty string"
        self.tokenizer: AutoTokenizer = tokenizer
        self.model = model
        self.batch_size = batch_size

        try:
            f = open(initial_suffix, 'r')
            self.initial_suffix = f.readline()
        except FileNotFoundError:
            self.initial_suffix = initial_suffix
    
        self.top_suffice = pickle.load(open(start_from_file,'rb')) if start_from_file else []
        if reuse_log and start_from_file: self.outfile_prefix = start_from_file[:-4]

        self.vocab_mask = None 
        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size
        print(vocab_size, self.tokenizer.vocab_size)
        self.special_token_mask = torch.zeros(vocab_size)
        
        for id in list(self.tokenizer.added_tokens_decoder.keys()): # mask special tokens
            try:
                self.special_token_mask[id] = 1
            except Exception as e:
                print(e)
                continue
        if mask_unrecognizable_unicode:
            useless_token_ids = []
            for i in range(vocab_size):
                try:
                    s = self.tokenizer.decode([i])
                except Exception as e:
                    print(e)
                    continue
                if '�' in s:
                    useless_token_ids.append(i)
            for id in useless_token_ids:
                self.special_token_mask[id] = 1
        if self.vocab == 'english':
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            # nlp = spacy.load("en_core_web_sm")
            # words_list = list(set(nlp.vocab.strings))
            self.vocab_mask = self.get_english_only_mask(words_list)
        elif self.vocab == 'non_english':
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            self.vocab_mask = self.get_non_english_mask(words_list)
        elif self.vocab == 'hybrid':
            words = (
                urllib3.PoolManager()
                .request("GET", "https://www.mit.edu/~ecprice/wordlist.10000")
                .data.decode("utf-8")
            )
            words_list = words.split("\n")
            self.english_mask = self.get_english_only_mask(words_list)
            self.non_english_mask = self.get_non_english_mask(words_list)         
               

    def get_english_only_mask(
        self,
        words_list: list[str],
    ) -> torch.Tensor:
        """
        Get english only tokens from the model's tokenizer
        """

        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        english_only_mask = torch.zeros(vocab_size)
        #english_only_mask = torch.ones(vocab_size)
        for word in tqdm(
            words_list, desc="Building non english only mask", total=len(words_list)
        ):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for word_id in word_ids:
                english_only_mask[word_id] = 1

        return english_only_mask

    def get_non_english_mask(
        self,
        words_list: list[str],
    ) -> torch.Tensor:
        """
        Get non english tokens from the model's tokenizer
        """

        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        english_only_mask = torch.ones(vocab_size)
        for word in tqdm(
            words_list, desc="Building non english only mask", total=len(words_list)
        ):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            for word_id in word_ids:
                english_only_mask[word_id] = 0

        return english_only_mask

    def gcg_gradients(
        self,
        sample: FullPrompt,
    ) -> torch.Tensor:
        """
        First part of GCG. Compute gradients of each suffix (or all) tokens in the sample for Greedy Coordinate Gradient

        Parameters
        ----------
            sample: FullPrompt
                Prompt to compute gradients for

        Returns
        -------
            torch.Tensor
                Gradients of the suffix tokens (suffix_len, vocab_size)
        """
        n_docs = len(sample.target_prefix_ids)
        
        if "vocab_size" not in self.model.__dict__:
            vocab_size = self.model.config.vocab_size
        else:
            vocab_size = self.model.vocab_size

        grads = torch.zeros(
            n_docs,
            sample.suffix_slice[0].stop - sample.suffix_slice[0].start,
            vocab_size,
            device=self.model.device,
        )

        for j in range(0, n_docs, self.batch_size):
            cur_batch_size = min(self.batch_size, n_docs - j)
            embs_list = []
            targets_list = []
            loss_slices = []

            one_hot_suffices = []

            for i in range(j, j + cur_batch_size):

                input_ids = sample.prompt_ids[i].to(self.model.device)
                target_docs = sample.target_prefix_ids[i]#: i + cur_batch_size]
                input_ids = torch.cat([input_ids, target_docs.to(input_ids.device)], dim=1)

                targets = sample.target_prefix_ids[i].to(self.model.device)
                targets_list.append(targets)
                loss_slices.append(slice(
                    sample.target_prefix_slice[i].start - 1,
                    sample.target_prefix_slice[i].stop - 1,
                ))

                model_embeddings = self.model.get_input_embeddings().weight

                one_hot_suffix = torch.zeros(
                    input_ids.shape[0],
                    sample.suffix_slice[i].stop - sample.suffix_slice[i].start,
                    model_embeddings.shape[0],
                    device=self.model.device,
                    dtype=model_embeddings.dtype,
                )


                one_hot_suffix.scatter_(
                    -1,
                    input_ids[:, sample.suffix_slice[i]].unsqueeze(-1),
                    1,
                )

                one_hot_suffices.append(one_hot_suffix)

                one_hot_suffices[-1].requires_grad = True

                suffix_embs = one_hot_suffices[-1] @ model_embeddings
                embeds = self.model.get_input_embeddings()(input_ids).detach()

                full_embs = torch.cat(
                    [
                        embeds[:, : sample.suffix_slice[i].start, :],
                        suffix_embs,
                        embeds[:, sample.suffix_slice[i].stop :, :],
                    ],
                    dim=1,
                )
                embs_list.append(full_embs)

            embs_list = self._batch_embs(embs_list)
            if 'glm' in self.model.name_or_path:
                logits = self.model(input_ids=torch.ones((embs_list.shape[0], embs_list.shape[1])).to(self.model.device), inputs_embeds=embs_list, past_key_values = self.kv_cache).logits # temporary workaround before glm fix this (currently it doesnt support inputembed only)
            else:
                logits = self.model(inputs_embeds=embs_list, past_key_values = self.kv_cache).logits

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            syn_fct = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=IGNORE_INDEX)

            targets_list = self._batch_targets(targets_list)

            logits = self._batch_logits(logits, loss_slices, targets_list.size(1))
            loss = loss_fct(logits.transpose(1, 2), targets_list)
            if self.syntax_weight > 0:
                syn_loss = 0
                syn_loss += syn_fct(logits.transpose(1, 2)[:,:,:self.syntax_prefix_length], targets_list[:,:self.syntax_prefix_length]) * self.syntax_weight
                syn_loss += syn_fct(logits.transpose(1, 2)[:,:,-self.syntax_suffix_length:], targets_list[:,-self.syntax_suffix_length:]) * self.syntax_weight
                syn_loss = syn_loss / (self.syntax_prefix_length + self.syntax_suffix_length) 
                loss += syn_loss

            loss.backward()
            for i in range(j, j + cur_batch_size):
                grads[i:i+1] = one_hot_suffices[i - j].grad.clone()

        return grads.mean(dim=0)

    @torch.no_grad()
    def proposal_loss(
        self,
        sample: FullPrompt,
        proposals: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run forward pass with the new proposals and get the loss.

        Parameters
        ----------
            sample: FullPrompt
                Prompt to compute loss for
            proposals: torch.Tensor
                Proposals to compute loss for (num_proposals, prompt_len)

        Returns
        -------
            torch.Tensor
                Loss for each proposal (num_proposals,)
        """

        proposal_losses = torch.zeros(
            proposals.shape[0],
            device=proposals.device,
        )
        n_docs = len(sample.target_prefix_ids)

        loss_fct = torch.nn.CrossEntropyLoss(
                    reduction="mean", ignore_index=IGNORE_INDEX
                )
        syn_fct =  torch.nn.CrossEntropyLoss(
                    reduction="sum", ignore_index=IGNORE_INDEX
                )
        
        for i in range(proposals.shape[0]):
            proposal = proposals[i]
            for k in range(0, n_docs, self.batch_size):
                cur_batch_size = min(self.batch_size, n_docs - k)
                embs_list = []

                for j in range(k, k + cur_batch_size):
                    full_proposal_input = torch.cat(
                        (
                            sample.prompt_ids[j][:, :sample.suffix_slice[j].start],
                            proposal.to(sample.target_prefix_ids[0].device).unsqueeze(0),
                            sample.prompt_ids[j][:, sample.suffix_slice[j].stop:],
                            sample.target_prefix_ids[j],
                        ),
                        dim=1,
                    )

                    proposal_embs = self.model.get_input_embeddings()(
                        full_proposal_input.to(self.model.device)
                    )
                    embs_list.append(proposal_embs)

                embs_list = self._batch_embs(embs_list)

                if 'glm' in self.model.name_or_path:
                    logits = self.model(input_ids=torch.ones((embs_list.shape[0], embs_list.shape[1])).to(self.model.device), inputs_embeds=embs_list, past_key_values = self.kv_cache).logits # temporary workaround before glm fix this (currently it doesnt support inputembed only)
                else:
                    logits = self.model(inputs_embeds=embs_list.to(self.model.device), past_key_values = self.kv_cache).logits


                for j in range(k, k+cur_batch_size):
                    loss_slice = slice(
                        sample.target_prefix_slice[j].start - 1,
                        sample.target_prefix_slice[j].stop - 1,
                    )

                    proposal_losses[i] += loss_fct(logits[j-k, loss_slice.start:loss_slice.stop].cpu(), sample.target_prefix_ids[j].squeeze() )
                    if self.syntax_weight > 0:
                        syn_loss = 0
                        syntax_logits = logits[j-k, loss_slice.start:loss_slice.start + self.syntax_prefix_length].cpu()
                        syntax_targets = sample.target_prefix_ids[j].squeeze()
                        syntax_targets = syntax_targets[:self.syntax_prefix_length]
                        syn_loss += syn_fct(syntax_logits, syntax_targets) * self.syntax_weight

                        syntax_logits = logits[j-k, loss_slice.stop - self.syntax_suffix_length:loss_slice.stop].cpu()
                        syntax_targets = sample.target_prefix_ids[j].squeeze()
                        syntax_targets = syntax_targets[-self.syntax_suffix_length:]
                        syn_loss += syn_fct(syntax_logits, syntax_targets) * self.syntax_weight
                        syn_loss = syn_loss / (self.syntax_suffix_length + self.syntax_prefix_length)
                        proposal_losses[i] += syn_loss

            proposal_losses[i] = proposal_losses[i] / n_docs
        return proposal_losses

    def gcg_replace_tok(
        self,
        sample: FullPrompt,
    ) -> tuple[torch.Tensor, float]:
        """
        This func implements part 2 of GCG. Now that we have the suffix tokens logits gradients w.r.t loss, we:
        For j = 0,...,total # of proposals
        1. Select the top-k logits for each suffix pos based on -grad of the logits
        2. For each proposal,
        3. Uniformly sample a random token in the top-k logits for replacement at position i
        4. Replace token i with the sampled token. Set this as proposal_j

        Run forward pass for all proposals, get the loss, and pick the proposal with the lowest loss.
        """

        # Compute gradients of the suffix tokens w.r.t the loss
        suffix_logits_grads = self.gcg_gradients(sample)

        suffix_logits_grads = suffix_logits_grads / suffix_logits_grads.norm(
            dim=-1, keepdim=True
        )
        
        suffix_logits_grads[:, self.special_token_mask == 1] = float("inf") # mask special tokens
        if self.vocab in ['english', 'non_english']:
            # clip all non-english tokens
            suffix_logits_grads[:, self.vocab_mask != 1] = float("inf")

            # Select the top-k logits for each suffix pos based on -grad of the logits
            _, top_k_suffix_indices = torch.topk(
                -suffix_logits_grads,
                k=self.k,
                dim=-1,
            )
        elif self.vocab == 'all_allow':
            _, top_k_suffix_indices = torch.topk(
                -suffix_logits_grads,
                k=self.k,
                dim=-1,
            )
        elif self.vocab == 'hybrid':
            suffix_logits_grads_tmp = suffix_logits_grads.detach().clone()
            suffix_logits_grads_tmp[:, self.english_mask != 1] = float("inf")

            # Select the top-k logits for each suffix pos based on -grad of the logits
            _, top_k_english_suffix_indices = torch.topk(
                -suffix_logits_grads_tmp,
                k=self.k // 2,
                dim=-1,
            )

            suffix_logits_grads_tmp = suffix_logits_grads.detach().clone()
            suffix_logits_grads_tmp[:, self.non_english_mask != 1] = float("inf")

            # Select the top-k logits for each suffix pos based on -grad of the logits
            _, top_k_non_english_suffix_indices = torch.topk(
                -suffix_logits_grads_tmp,
                k=self.k // 2,
                dim=-1,
            )

            #top_k_suffix_logits_grads = torch.cat([top_k_english_suffix_logits_grads, top_k_non_english_suffix_logits_grads], dim = 1)
            top_k_suffix_indices = torch.cat([top_k_english_suffix_indices, top_k_non_english_suffix_indices], dim = 1)

        self.total_proposals = self.num_proposals * top_k_suffix_indices.shape[0]
        proposals = sample.prompt_ids[0][:,sample.suffix_slice[0]].repeat(self.total_proposals, 1).to(
            top_k_suffix_indices.device
        )

        for i in range(self.total_proposals):
            proposal_suffix_ids = proposals[i]
            proposal_suffix_ids[i % suffix_logits_grads.shape[0]] = torch.gather(
                top_k_suffix_indices[i % suffix_logits_grads.shape[0]],
                0,
                torch.randint(
                    0,
                    top_k_suffix_indices.shape[-1],
                    (1,),
                    device=top_k_suffix_indices.device,
                ),
            )
            proposals[i] = proposal_suffix_ids

        if self.ppl_preference > 1:
            perplexity_losses = []
            loss_fct = torch.nn.CrossEntropyLoss(
                        reduction="mean", ignore_index=IGNORE_INDEX
                    )
            for i in range(len(proposals)):
                logits = self.model(input_ids=proposals[i].unsqueeze(0)).logits
                perplexity_losses.append(loss_fct(logits.squeeze(), proposals[i]))

        # Now compute the loss for each proposal, and pick the next candidate as the lowest one
        with torch.no_grad():
            proposal_losses = self.proposal_loss(sample, proposals)

        best_proposal_idx = proposal_losses.argmin() # first to change
        best_proposal = proposals[best_proposal_idx]

        # Now update the sample with the new suffix
        if self.temperature > 0:
            tmp_proposal_losses = torch.max(proposal_losses) - proposal_losses
            tmp_proposal_losses = tmp_proposal_losses / self.temperature
            proposal_logits = torch.exp(tmp_proposal_losses) / torch.sum(torch.exp(tmp_proposal_losses))
            best_proposal_idx = proposal_logits.multinomial(num_samples=1, replacement=True)[0].item()
            best_proposal = proposals[best_proposal_idx]
            return best_proposal, proposal_losses[best_proposal_idx].cpu().item() 
        else:
            if self.ppl_preference > 1:
                best_id = 0
                highest_ppl = 0
                for i in range(len(proposal_losses)):
                    loss = proposal_losses[i].item()
                    if loss < proposal_losses.min().item() * self.ppl_preference:
                        if perplexity_losses[i].item() > highest_ppl:
                            highest_ppl = perplexity_losses[i].item()
                            best_id = i
                best_proposal = proposals[best_id]
                return best_proposal, proposal_losses[best_id].item()
            else:
                best_proposal_idx = proposal_losses.argmin()
                best_proposal = proposals[best_proposal_idx]
                return best_proposal, proposal_losses.min().item()

    def load_dataset(
        self,
        dataset: str | list,
    ) -> None:
        """
        Load a dataset from a json file

        Parameters
        ----------
            dataset: str | list
                Path to the dataset file or the dataset itself
        """
        data = []
        common_syntax_prefix = None
        common_syntax_suffix = None
        common_prefix = None

        train_docs = []
        train_prompt_ids = []
        train_suffix_slices = []
        train_target_prefix_slices = []

        for entry in dataset:
            objective_token_id = self.tokenizer.encode(entry['objective'], return_tensors="pt", add_special_tokens=False, truncation=True)

            if common_syntax_prefix is None:
                common_syntax_prefix = torch.clone(objective_token_id)
                common_syntax_suffix = torch.clone(objective_token_id)
            else:
                for i in range(common_syntax_prefix.size(1)):
                    if objective_token_id[0, i].item() == common_syntax_prefix[0, i].item():
                        continue
                    else:
                        common_syntax_prefix = common_syntax_prefix[:, :i]
                        break
                
                for i in range(common_syntax_suffix.size(1) - 1, 0, -1):
                    if objective_token_id[0, i-common_syntax_suffix.size(1)].item() == common_syntax_suffix[0, i].item():
                        continue
                    else:
                        common_syntax_suffix = common_syntax_suffix[:, i:]
                        break

            train_docs.append(objective_token_id)

            context_prompt, train_slice = utils.build_context_prompt(
                self.model.config.name_or_path, entry['conversations'], self.initial_suffix, self.tokenizer
            )

            if self.top_suffice: # start_from_file
                context_prompt[:, train_slice] = heapq.nlargest(1, self.top_suffice)[0][3] # 0 to get the last entry i.e. the best suffix; 3 -> tokens of the suffix


            if common_prefix is None:
                common_prefix = torch.clone(context_prompt)
            else:
                for i in range(common_prefix.size(1)):
                    if context_prompt[0, i].item() == common_prefix[0, i].item():
                        continue
                    else:
                        common_prefix = common_prefix[:, :i]
                        break

            train_prompt_ids.append(context_prompt)
            train_suffix_slices.append(train_slice)

            train_target_prefix_slices.append(slice(
                train_prompt_ids[-1].shape[-1],
                train_prompt_ids[-1].shape[-1] + objective_token_id.shape[-1],
            ) )

            context_prompt, train_slice = utils.build_context_prompt(
                self.model.config.name_or_path, entry['conversations'], self.initial_suffix, self.tokenizer
            )

        self.syntax_prefix_length = len(common_syntax_prefix[0])
        self.syntax_suffix_length = len(common_syntax_suffix[0])
        print(f"The common syntax prefix is {self.tokenizer.decode(common_syntax_prefix[0])} which is the first {self.syntax_prefix_length} tokens!")
        print(f"The common syntax suffix is {self.tokenizer.decode(common_syntax_suffix[0])} which is the last {self.syntax_suffix_length} tokens!")

        len_prefix = common_prefix.size(1)
        print(f"The cached system prompt is {self.tokenizer.decode(common_prefix[0])} which has {len_prefix} tokens")
        for i in range(len(train_prompt_ids)):
            train_prompt_ids[i] = train_prompt_ids[i][:, len_prefix:]
        for i in range(len(train_suffix_slices)):
            train_suffix_slices[i] = slice(train_suffix_slices[i].start - len_prefix, train_suffix_slices[i].stop - len_prefix)
        for i in range(len(train_target_prefix_slices)):
            train_target_prefix_slices[i] = slice(train_target_prefix_slices[i].start - len_prefix, train_target_prefix_slices[i].stop - len_prefix)
        self.common_prefix = common_prefix

        dic = {
            "input_token_ids": train_prompt_ids,
            "adversarial_slice": train_suffix_slices,
            "output_token_ids": train_docs,
            "output_slice": train_target_prefix_slices
        }
        
        self.dataset = FullPrompt(
                    prompt_ids=train_prompt_ids,
                    suffix_slice=train_suffix_slices,
                    target_prefix_slice=train_target_prefix_slices,
                    target_prefix_ids=train_docs,
                    prompt_ident=0,
                )
        self.datasets = data
        pickle.dump(dic, open('./raw_data_tokens.pkl', 'wb'))

    def train(
        self,
        trial: int,
    ) -> None:
        """
        Parameters
        ----------
            trial: int
                Trial number
        """

        train_sample = self.dataset
        prompt_id = train_sample.prompt_ident

        pbar = tqdm(range(self.num_epochs), total=self.num_epochs)
        restart_tracker = 0

        for param in self.model.parameters():
            param.requires_grad = False
        
        self.common_prefix = self.common_prefix.repeat(self.batch_size, 1)
        self.kv_cache = self.model(self.common_prefix.to(self.model.device), use_cache = True).past_key_values

        with open(self.outfile_prefix+f".log", "a", encoding='utf-8') as f:
            f.write(f"|------------TRIAL {trial}-------------|")

        if not self.top_suffice:
            init_id = 0
            
            with torch.no_grad():
                initial_suffix_token = train_sample.prompt_ids[0][0][train_sample.suffix_slice[0]].unsqueeze(0).to(self.model.device)
                loss_0 = self.proposal_loss(train_sample, initial_suffix_token)[0]

            heapq.heappush(self.top_suffice, [-loss_0, self.initial_suffix, -1, train_sample.prompt_ids[0][0][train_sample.suffix_slice[0]]])

            with open(self.outfile_prefix+".log", "a", encoding='utf-8') as f:
                f.write(f"""
                Initial Prompt: {self.initial_suffix}, 
                Length: {train_sample.suffix_slice[0].stop - train_sample.suffix_slice[0].start} tokens, 
                Loss: {loss_0:.2f}\n"""
                )
        else:
            best_suffix = heapq.nlargest(1, self.top_suffice)[0]
            init_id = best_suffix[2] + 1 # +1 to make sure the new epoch marked from the next id
            with open(self.outfile_prefix+".log", "a", encoding='utf-8') as f:
                f.write(f"""Resume from stored state
                    Initial Prompt: {best_suffix[1]}, 
                    Length: {len(best_suffix[3])} tokens, 
                    Saved Loss: {-best_suffix[0]}\n
                    Actual Loss against entire testset: {-self.proposal_loss(train_sample, best_suffix[3].unsqueeze(0))}
                """
                )
            
        try:
            for i in pbar:
                current_sample = self.random_choose(train_sample, self.subset_size)
                best_proposal, loss = self.gcg_replace_tok(current_sample)
                train_sample.update_suffix(best_proposal)
                suf = self.tokenizer.decode(best_proposal)  # current suffix

                if len(self.top_suffice) < 100:
                    heapq.heappush(self.top_suffice, [-loss, suf, i+init_id, best_proposal]) # minus loss because python heap pq is ascending.
                else:                      
                    heapq.heappushpop(self.top_suffice, [-loss, suf, i+init_id, best_proposal])
                
                best_loss, _, best_at, _ = heapq.nlargest(1, self.top_suffice)[0]

                with open(self.outfile_prefix+".log", "a", encoding='utf-8') as f:
                    f.write(f"Epoch: {i+init_id}; Suffix: {suf}\nloss:{loss:.2f};\n---- Best loss so far: {-best_loss} at epoch {best_at}. Average Epoch Speed: {pbar.format_dict['rate']}s\n")

                with open(self.outfile_prefix+'.pkl', 'wb') as f:
                    pickle.dump(self.top_suffice, f)

                pbar.set_description(
                    f"Epoch loss:{loss:.2f};Best loss so far: {-best_loss} at epoch {best_at}."
                )

                if self.autorestart > 0 and i+init_id-best_at > self.autorestart:
                    if i + init_id - restart_tracker < self.autorestart:
                        continue
                    if restart_tracker > 0:
                        return # go to next trail
                    new_start = self.top_suffice[0]
                    restart_tracker = i + init_id
                    train_sample.update_suffix(new_start[3])
                    
                    with open(self.outfile_prefix+f".log", "a", encoding='utf-8') as f:
                        f.write(f"Autorestart after not seeing progress in {self.autorestart} epochs. Picked\n{new_start[1]}\nfrom epoch {new_start[2]} with loss {new_start[0]} as the new start.")

        except KeyboardInterrupt:
            print('I am keyboard interrupted!!!!')
            with open(self.outfile_prefix+f".pkl", 'wb') as f:
                pickle.dump(self.top_suffice, f)
            print('state saved before exiting.')
            exit(130)

        return {
            "prompt_id": prompt_id,
            "trial": trial,
        }


    def random_choose(
        self,
        sample: FullPrompt,
        subset_size: int
    ) -> FullPrompt:
        """
        Randomly choose a subset of samples for GCG
        """
                
        subset_size = min(len(sample.prompt_ids), subset_size)

        idx = list(range(len(sample.prompt_ids)))
        sampled_idx = random.sample(idx, subset_size)

        sampled_idx.sort(key=lambda x: len(sample.prompt_ids[x][0]))

        sampled_prompt_ids = [sample.prompt_ids[i] for i in sampled_idx]
        sampled_suffix = [sample.suffix_slice[i] for i in sampled_idx]
        sampled_prefix = [sample.target_prefix_slice[i] for i in sampled_idx]
        sampled_predix_ids = [sample.target_prefix_ids[i] for i in sampled_idx]

        return FullPrompt(
            prompt_ids=sampled_prompt_ids,
            suffix_slice=sampled_suffix,
            target_prefix_slice=sampled_prefix,
            target_prefix_ids=sampled_predix_ids,
            prompt_ident=0,
        )

    """
    Helper functions to batchify examples for (minor) acceleration
    """

    def _batch_embs(self, embs, max_len = None):
        len_list = [embs[i].size(1) for i in range(len(embs))]
        if not max_len:
            max_len = max(len_list)

        for i in range(len(embs)):
            padding = torch.zeros((1, max_len - len_list[i], embs[i].size(2)) ).to(embs[i].device).type(embs[i].dtype)
            embs[i] = torch.cat([embs[i], padding], dim = 1)
        embs = torch.cat(embs, dim = 0)
        return embs

    def _batch_targets(self, targets, max_len = None):
        len_list = [targets[i].size(1) for i in range(len(targets))]
        if not max_len:
            max_len = max(len_list)

        for i in range(len(targets)):
            padding = torch.ones((1, max_len - targets[i].size(1))).to(targets[i].device).type(targets[i].dtype) * IGNORE_INDEX
            targets[i] = torch.cat([targets[i], padding], dim = 1)
        targets = torch.cat(targets, dim = 0)
        return targets


    def _batch_logits(self, logits, slices, len_targets):
        logits = torch.cat([logits, torch.zeros_like(logits)], dim = 1)
        logits_list = []
        for i in range(len(logits)):
            logits_list.append(logits[i:i+1, slices[i].start:slices[i].start+len_targets])
        logits_list = torch.cat(logits_list, dim = 0)
        return logits_list