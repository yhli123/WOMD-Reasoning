# [ICML 2025] WOMD-Reasoning

*Yiheng Li, Chongjian Ge, Chenran Li, Chenfeng Xu, Masayoshi Tomizuka, Chen Tang, Mingyu Ding, Wei Zhan*

Official Repo for WOMD-Reasoning: A Large-Scale Language Dataset for Interaction and Driving Intentions Reasoning, an ICML 2025 paper. Waymo Open Motion Dataset and WOMD are trademarks of Waymo LLC, and are used here by permission.

*Note: The dataset has been moved to [Waymo Official Website](https://waymo.com/open/download) for viewing and downloading.*

## Overview
WOMD-Reasoning is a language annotation dataset built on the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/data/motion/), with a focus on describing and reasoning interactions and intentions in driving scenarios. It presents by far the largest Q&A dataset on real-world driving scenarios, with around 3 million Q&As covering various topics of autonomous driving from map descriptions, motion status descriptions, to narratives and analyses of agents’ interactions, behaviors, and intentions.

## Data Structure
### File Description
The dataset is separated into two main subsets:
- `training.tar.gz`: The language annotations on WOMD training part.
- `validation_interactive`: The language annotations on WOMD validation-interactive part.
- 'Prompts': The whole set of ChatGPT prompts used to building WOMD-R.
- 'Motion_Data_2_Raw_Language_Translator.py': The program to convert WOMD motion data into raw language.

Both sets are compressed in tar.gz format.

### Data Fields
The dataset is provided in JSON format after extraction. An example structure and the meaning of each part are shown below:

```json
{
"sid": "WOMD Scene ID", 
"ego": "WOMD ID for ego agent", 
"cur_time": "The time (seconds) marked as current moment", 
"future_time": "The future time period (seconds) for interaction & intentions", 
"rel_id": ["List of WOMD IDs for the surrounding agents"],
"rel_qa_id": ["List of IDs for surrounding agents in the Q&As"], 
"env_q": ["Question 1 in map environments", "Question 2 in map environments"], 
"env_a": ["Answer 1 in map environments", "Answer 2 in map environments"], 
"ego_q": ["Question 1 in ego agent's motion status", "Question 2 in ego agent's motion status"], 
"ego_a": ["Answer 1 in ego agent's motion status", "Answer 2 in ego agent's motion status"], 
"sur_q": ["Question 1 in surrounding agent's motion status", "Question 2 in surrounding agent's motion status"], 
"sur_a": ["Answer 1 in surrounding agent's motion status", "Answer 2 in surrounding agent's motion status"], 
"int_q": ["Question 1 in interactions and intentions", "Question 2 in interactions and intentions"], 
"int_a": ["Answer 1 in interactions and intentions", "Answer 2 in interactions and intentions"],
}
```

Note that to avoid over-fitting, we alter the real IDs in the WOMD for agents with an alternate Q&A ID. The Q&A IDs [0-100) indicate vehicles, [100-200) indicates bicycles and [200-300) indiccates pedestrians.

### Data Preprocessing

The data preprocessing is straightforward by simply unzipping the tar.gz file to obtain the JSON file.

```python
  tar -xzvf training.tar.gz
  tar -xzvf validation_interactive.tar.gz
```

## License

In addition to the [Terms](https://waymo.com/open/terms/) from [WOMD Website](https://waymo.com/open), the language annotation dataset is subject to the LICENSE associated with the files.
