# Imprompter: Tricking LLM Agents into Improper Tool Use

This is the codebase of `imprompter`. It provides essential components to reproduce and test the attack presented in the [paper](https://arxiv.org/abs/2410.14923).  You may create your own attack on top of it as well.

A screencast showing how an attacker can exfiltrate the user's PII in real world LLM product ([Mistral LeChat](https://chat.mistral.ai/chat)) with our adversarial prompt:

![img](docs/attack_screenshot_annotated.png)

More video demos can be found on our [website](https://imprompter.ai). **Meanwhile, big thanks to Matt Burges from WIRED and Simon Willison for writing cool stories ([WIRED](https://www.wired.com/story/ai-imprompter-malware-llm/), [Simon's Blog](https://simonwillison.net/2024/Oct/22/imprompter/)) covering this project!**

## Setup

Setup python environment with `pip install .` or `pdm install` ([pdm](https://github.com/pdm-project/pdm)). We recommend using virtual environment (e.g. `conda` with `pdm venv`).

For `GLM4-9b` and `Mistral-Nemo-12B` a 48GB VRAM GPU is required. For `Llama3.1-70b` 3x 80GB VRAM is required.

## Configuration

There are two config files that need potential attention before you run the algorithm
- `./configs/model_path_config.json` defines the path of the huggingface model on your system. You most likely need to modify this accordingly.
- `./configs/device_map_config.json` configures layer mapping for loading the models on multi-gpu. We show case our configuration for loading LLama-3.1-70B on 3x Nvidia A100 80G GPUs. You may need to adjust this accordingly for your computational environments.

## Run Optimization

Follow the example execution scripts e.g. `./scripts/T*.sh`. The explainations of each argument can be found in Section 4 of our [paper](https://arxiv.org/abs/2410.14923).

The optimization program will generate results in `.pkl` files and logs in the `./results` folder. The pickle file updates every step during the execution and always stores the current top 100 adversarial prompts (with lowest loss). It is structured as a min heap, where the top of it is the prompt with the lowest loss. Each element of the heap is a tuple of `(<loss>, <adversarial prompt in string>, <optimization iteration>, <adversarial prompt in tokens>)`. You can always restart from an existing pickle file by adding `--start_from_file <path_to_pickle>` arguments to its original execution script.

## Evaluation

Evaluation is done through `evaluation.ipynb`. Follow the detailed instructions there to  generations against testing dataset, computation of metrics, etc. 


One special case is the PII prec/recall metrics. They are computed standalone with `pii_metric.py`. Note that `--verbose` gives your full PII details of each conversation entry for debugging and `--web` should be added when the results are obtained from real products on the web. 

Example usage (non web result i.e. local test):

`python pii_metric.py --data_path datasets/testing/pii_conversations_rest25_gt.json --pred_path evaluations/local_evaluations/T11.json`

Example usage (web result i.e. real product test):

`python pii_metric.py --data_path datasets/testing/pii_conversations_rest25_gt.json --pred_path evaluations/product_evaluations/N6_lechat.json --web --verbose`

## Browser Automation

We use Selenium to automate the testing process on real products (Mistral LeChat and ChatGLM). We provide the code in `browser_automation` directory. Note we have only tested this on a desktop environment on Windows 10 and 11. It is supposed to work also on Linux/MacOS but not guaranteed. Might need some small tweaks.  

Example usage:
`python browser_automation/main.py --target chatglm --browser chrome --output_dir test --dataset datasets/pii_conversations_rest25_gt.json --prompt_pkl results/T12.pkl --prompt_idx 1`

- `--target` specifies the product, right now we support `chatglm` and `mistral` two options. 
- `--browser` defines the browser to be using, you should either use `chrome` or `edge`.  
- `--dataset` points to the conversation dataset to test with
- `--prompt_pkl` refers the pkl file to read prompt from and `--prompt_idx` defines the ordered index of the prompt to use from the pkl. Alternatively, one may define the prompt in `main.py` directly and do not provide these two options.  

## Reproducibility

We provide all the scripts (`./scripts`) and datasets (`./datasets`) to obtain the prompts (T1-T12) we present in the [paper](https://arxiv.org/abs/2410.14923). Moreover, we also provide the pkl result file (`./results`) for each of the prompt as long as we still keep a copy and the evaluation result of them (`./evaluations`) obtained through `evaluation.ipynb`. Note that for PII Exfiltration attack, the training and testing datasets contain real-world PII. Even though they are obtained from the public [WildChat](https://wildchat.allen.ai/) Dataset, we decide to not make them directly public for privacy concerns. We provide a single entry subset of these datasets at `./datasets/testing/pii_conversations_rest25_gt_example.json` for your reference. Please contact us to request the complete version of these two datasets. 

## Disclosure and Impact

We initiated disclosure to Mistral and ChatGLM team on Sep 9, 2024, and Sep 18, 2024, respectively. Mistral security team members responded promptly and acknowledged the vulnerability as a **medium-severity issue**. They fixed the data exfiltration by disabling markdown rendering of external images on Sep 13, 2024 (find the acknowledgement in [Mistral changelog](https://docs.mistral.ai/getting-started/changelog/)). We confirmed that the fix works. ChatGLM team responded to us on Oct 18, 2024 after multiple communication attempts through various channels and stated that they have begun working on it.

## Citation

Please consider citing our [paper](https://arxiv.org/abs/2410.14923) if you find this work valuable.

```tex
@misc{fu2024impromptertrickingllmagents,
      title={Imprompter: Tricking LLM Agents into Improper Tool Use}, 
      author={Xiaohan Fu and Shuheng Li and Zihan Wang and Yihao Liu and Rajesh K. Gupta and Taylor Berg-Kirkpatrick and Earlence Fernandes},
      year={2024},
      eprint={2410.14923},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2410.14923}, 
}
```
