
<div align="center">
<h1><a href="https://arxiv.org/pdf/2604.05995" style="color:#68edcb">The Model Agreed, But Didn’t Learn: Diagnosing Surface Compliance in Large Language Models</a></h1>

[![arXiv](https://img.shields.io/badge/arXiv-2603.16654-b31b1b.svg?style=plastic)](https://arxiv.org/pdf/2604.05995)
</div>




## Environment Setup

```bash
conda create -n ultraedit python=3.10
conda activate ultraedit
pip install torch==2.3.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```


## Eval

Run evaluation with a vanilla model and an edited model:

```bash
python eval.py \
  --vanilla_model Qwen/Qwen2.5-7B-Instruct \
  --edited_model PUT_THE_EDITED_MODEL_HERE \
  --data_path zsre_966.json \
  --metrics exact_match_tf,exact_match_wo_tf,likelihood_margin,sa_mcq
```

Available metrics:

```text
exact_match_tf
exact_match_wo_tf
llm_as_judge
likelihood_margin
sa_mcq
all
```

To run `LLM-as-judge`, provide an OpenAI-compatible API key:

```bash
python eval.py \
  --vanilla_model Qwen/Qwen2.5-7B-Instruct \
  --edited_model PUT_THE_EDITED_MODEL_HERE \
  --metrics llm_as_judge \
  --api_key YOUR_API_KEY \
  --judge_workers 100
```

The output is saved to `eval_compare_results.json` by default. Use `--output_path` to change it.


## Contact

For any inquiries, please reach out at **peettherapynoys@gmail.com**



## Citation

If you find SA-MCQ useful for your research and applications, please cite:

```bibtex
@inproceedings{gu2026modelagreeddidntlearn,
  title={The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models}, 
  author={Xiaojie Gu and Ziying Huang and Weicong Hong and Jian Xie and Renze Lou and Kai Zhang},
  booktitle={Findings of ACL},
  year={2026}
}
```
