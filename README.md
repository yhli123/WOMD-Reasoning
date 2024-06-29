# WOMD-Reasoning

*Yiheng Li, Chongjian Ge, Chenran Li, Chenfeng Xu, Masayoshi Tomizuka, Chen Tang, Mingyu Ding, Wei Zhan*

Official Github Repo for WOMD-Reasoning: A Large-Scale Language Dataset for Interaction and Driving Intentions Reasoning. Waymo Open Motion Dataset and WOMD are trademarks of Waymo LLC, and are used here by permission.

## Introduction
WOMD-Reasoning is by far the largest real-world language dataset for autonomous driving containing description and reasoning labels for diverse interaction scenarios.

## Data Source
The dataset is built based on [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/data/motion/). Please refer to WOMD for the motion data. 
You must register at [WOMD Website](https://waymo.com/open) and agree to their [Terms](https://waymo.com/open/terms/) in addition to the terms of THIS language annotation dataset.

## Data Structure
### File Description
The dataset is separated into two main sets:
- `training.tar.gz`: The language annotations on WOMD training part.
- `validation_interactive`: The language annotations on WOMD validation-interactive part.

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

The dataset is straightforward to preprocess. Simply unzip the tar.gz file to obtain the JSON file.

```python
  tar -xzvf training.tar.gz
  tar -xzvf validation_interactive.tar.gz
```

### License

ATTENTION! You must register at [WOMD Website](https://waymo.com/open) and agree to their [Terms](https://waymo.com/open/terms/) to use THIS language annotation dataset.
IN ADDITION, This language annotation dataset as well as this github repo is under [this license](https://github.com/yhli123/WOMD-Reasoning/blob/main/LICENSE).

### Contribution

To contribute to the improvement of this dataset, please provide feedback or suggestions.
You can contact [Yiheng Li](mailto:yhli@berkeley.edu) for this.
