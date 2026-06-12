import copy
import itertools

from transformers import GemmaTokenizerFast

from arxiv.methods.preprocessing.base import Preprocessor

class Seq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences = self.tokenizer(
            [self.model_args.prompt_prefix + source for source in features["input"]], 
            max_length=self.model_args.max_input_length,
            truncation=True
        )["input_ids"]
        labels = self.tokenizer(
            features["output"], 
            max_length=self.model_args.max_input_length,
            truncation=True
        )["input_ids"]

        return sequences, labels


class CausalSeq2SeqPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences, labels, sequences_no_labels = [], [], []
        output_key = "output" if "output" in features else "label"
        for source, target in zip(features["input"], features[output_key]):
            if "qwen" in self.model_args.model_name_or_path.lower():
                sys_id = "assistant"
            elif "llama" in self.model_args.model_name_or_path.lower():
                sys_id = "model"
            else:
                sys_id = "system"
            if source is not None and target is not None:
                source = self.tokenizer.batch_decode(
                    [self.tokenizer(
                        source,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=self.model_args.max_input_length
                    )["input_ids"]],
                    skip_special_tokens=True
                )[0]
                try:
                    messages = []
                    if self.model_args.system_prompt != "":
                        messages.append({
                            "role": "system",
                            "content": self.model_args.system_prompt,
                        })
                    messages.append({
                        "role": "user",
                        "content": self.model_args.prompt_prefix + " " + source
                    })
                    input_no_label = self.tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=not "llama" in self.model_args.model_name_or_path, 
                        date_string = "25 Dec 2025", 
                        enable_thinking=False)
                    messages.append({
                        "role": sys_id,
                        "content": target
                    })
                    input_label = self.tokenizer.apply_chat_template(
                        messages,
                        date_string = "25 Dec 2025", 
                        enable_thinking=False
                    )
                    tokenized_target = input_label
                except ValueError:
                    # GPT2 trained from scratch
                    input_no_label = self.tokenizer(source, add_special_tokens=False)["input_ids"] 
                    input_label = self.tokenizer(source + target, add_special_tokens=False)["input_ids"] + [self.tokenizer.eos_token_id]

            sequences.append(input_label)
            sequences_no_labels.append(input_no_label)
            labels.append(self.tokenizer(target, add_special_tokens=False)["input_ids"])

        return sequences_no_labels, sequences if train else labels


class MathDialPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences, labels = [], []
        for turns, response, dialog_act, student_profile in zip(
            features["turns"], 
            features["output"], 
            features["dialog_act"], 
            features["student_profile"]
        ):
            if "TutorRL" in self.model_args.model_name_or_path:
                conversation = [{
                    "role": "assistant" if turn["role"] == "teacher" else "user",
                    "content": turn["text"]
                } for turn in turns]
            else:
                turns = [{
                    "role": "assistant" if turn["role"] == "teacher" else "user",
                    "content": turn["text"]
                } for turn in turns]
                formatted_turns = "Continue the following dialog as a teacher:\n"
                for turn in turns:
                    formatted_turns += f"{turn['role']}: {turn['content']}\n"
                conversation = [
                    {
                        "role": "system",
                        "content": "You are tasked with being a teacher and helping a student with a math problem.\n\nYou must not reveal the answer to the problem to the student at any point in time.\nYour task is to guide the student to have a complete understanding of the problem.\nEven if the student is already able to solve the problem, you should help them understand and improve the solution so that they get as high of a grade as possible.\n\nIf possible, do not respond with overly long responses to the student."
                    },
                    {
                        "role": "user",
                        "content": formatted_turns
                    }
                ]
            conversation = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            sequences.append(conversation)
            labels.append(response)

        sequences = self.tokenizer(sequences)["input_ids"]
        labels = self.tokenizer(labels)["input_ids"]

        return sequences, labels

class ReviewDirectPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences, labels = [], []
        for paper, reviews in zip(
            features["paper"], 
            features["reviews"], 
        ):  
            conversation = [{
                "role": "system",
                "content": self.model_args.system_prompt
            }]
            try:
                conversation_system_only = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    truncation=False, 
                    enable_thinking=False
                )
            except:
                conversation_system_only = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    truncation=False, 
                )
            tokenized_paper = self.tokenizer(f"## Paper:\n{paper}", add_special_tokens=False)["input_ids"]
            if len(tokenized_paper) > self.model_args.max_input_length-self.model_args.generation_max_len-len(conversation_system_only) - 10:
                tokenized_paper = tokenized_paper[:self.model_args.max_input_length-self.model_args.generation_max_len-len(conversation_system_only) - 20] # 20 is a buffer as there are some fluctuations in tokenization depeending on the full convo context
                paper = self.tokenizer.batch_decode([tokenized_paper])[0]
            conversation.append({
                "role": "user",
                "content": f"## Paper:\n{paper}"
            })
            try:
                conversation = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    truncation=False, 
                    enable_thinking=False
                )
            except:
                conversation = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    truncation=False, 
                )
            sequences.append(conversation)
            labels.append(reviews[0]["text"]) # convenience, not needed for search

        labels = self.tokenizer(labels)["input_ids"]
        return sequences, labels


class DialogActPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences, labels = [], []
        choices = self.model_args.possible_acts.split(";")
        print(choices)
        for idx, choice in enumerate(choices):
            choices[idx] = self.tokenizer(choice, add_special_tokens=False)["input_ids"]
        for turns, incorrect_solution, dialog_act in zip(features["turns"], features["incorrect_solution"], features["dialog_act"]):
            formatted_turns = "## Conversation:\n"
            for turn in turns:
                formatted_turns += f"{turn['role']}: {turn['text']}\n"
            conversation = [{
                "role": "system",
                "content": self.model_args.system_prompt
            }]
            conversation.append({
                "role": "user",
                "content": f"{formatted_turns} # Incorrect Solution: {incorrect_solution}"
            })

            conversation = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=True, 
                add_generation_prompt=True, 
                max_length=self.model_args.max_input_length-self.model_args.generation_max_len, 
                truncation=False, 
                enable_thinking=False
            )
            sequences.append(conversation)
            labels.append(self.tokenizer(dialog_act, add_special_tokens=False)["input_ids"])

        return sequences, labels


class ReviewScorePreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        sequences, labels = [], []
        choices = self.model_args.possible_scores.split(";")
        for idx, choice in enumerate(choices):
            choices[idx] = self.tokenizer(choice, add_special_tokens=False)["input_ids"]
        for paper, reviews in zip(
            features["paper"], 
            features["reviews"], 
        ):  
            for review in reviews:
                conversation = [{
                    "role": "system",
                    "content": self.model_args.system_prompt
                }]
                conversation.append({
                    "role": "user",
                    "content": f"## Review:\n{review['text']}"
                })
                conversation = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    max_length=self.model_args.max_input_length-self.model_args.generation_max_len-10, 
                    truncation=True, 
                    enable_thinking=False
                )
                sequences.append(conversation)
                labels.append([float(review["score"])])

        # labels = self.tokenizer(labels)["input_ids"]

        return sequences, labels

class ReviewScoreNoOraclePreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        import json
        sequences, labels = [], []
        choices = self.model_args.possible_scores.split(";")
        for idx, choice in enumerate(choices):
            choices[idx] = self.tokenizer(choice, add_special_tokens=False)["input_ids"]
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        for review in reviews:
            # try:
            conversation = [{
                "role": "system",
                "content": self.model_args.system_prompt
            }]
            conversation.append({
                "role": "user",
                "content": f"## Review:\n{review['sequence'].replace("json", "").replace("{", "").replace("}", "")}"
            })
            conversation = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=True, 
                add_generation_prompt=True, 
                max_length=self.model_args.max_input_length-self.model_args.generation_max_len-10, 
                truncation=True, 
                enable_thinking=False
            )
            sequences.append(conversation) #+ choice)
            labels.append([0.0])

        return sequences, labels

class MathDialQualityProcessor(Preprocessor):

    def preprocess(self, features, train=True):
        import json
        sequences, labels = [], []
        choices = self.model_args.possible_acts.split(";")
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        for turn in reviews:
            input_sequence = turn["input"].split("\n\n\n")[-1].replace("assistant", "teacher").replace("user", "student")
            for choice in choices:
                formatted_turns = f"## Conversation:\n{input_sequence}{turn['sequence']}\n## Tutoring Strategy:\n{choice}"
                conversation = [{
                    "role": "system",
                    "content": self.model_args.system_prompt
                }]
                conversation.append({
                    "role": "user",
                    "content": formatted_turns
                })

                conversation = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    max_length=self.model_args.max_input_length-self.model_args.generation_max_len, 
                    truncation=False, 
                    enable_thinking=False
                )
                sequences.append(conversation) 
                labels.append([0])

        return sequences, labels


class ReviewScoreConsistencyProcessor(Preprocessor):

    def preprocess(self, features, train=True):
        import json
        from tqdm import tqdm
        sequences, labels = [], []
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        num_possible_scores = len(self.model_args.possible_scores.split(";"))
        for review in tqdm(reviews):
            review = review["sequence"]
            for score in range(num_possible_scores):
                conversation = [{
                    "role": "system",
                    "content": self.model_args.system_prompt
                }]
                conversation.append({
                    "role": "user",
                    "content": f"## Review:\n {review} ## Score: {score}"
                })
                conversation = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=True, 
                    add_generation_prompt=True, 
                    max_length=self.model_args.max_input_length-self.model_args.generation_max_len, 
                    truncation=False, 
                    enable_thinking=False
                )
                sequences.append(conversation) 
                labels.append([.0]) # we don't have ground-truth labels here

        return sequences, labels


class ReviewQualityProcessor(Preprocessor):

    def preprocess(self, features, train=True):
        import json
        from tqdm import tqdm
        sequences, labels = [], []
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        for review in tqdm(reviews):
            review = review["sequence"]
            conversation = [{
                "role": "system",
                "content": self.model_args.system_prompt
            }]
            conversation.append({
                "role": "user",
                "content": f"## Review:\n {review}"
            })
            conversation = self.tokenizer.apply_chat_template(
                conversation, 
                tokenize=True, 
                add_generation_prompt=True, 
                max_length=self.model_args.max_input_length-self.model_args.generation_max_len, 
                truncation=False, 
                enable_thinking=False
            )
            sequences.append(conversation) 
            labels.append([.0]) # we don't have ground-truth labels here

        return sequences, labels


class ReviewMBRPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        import json
        from tqdm import tqdm
        sequences, labels = [], []
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        for review in tqdm(reviews):
            review = review["sequence"]
            conversation = self.tokenizer(
                review, 
                truncation=False, 
            )["input_ids"]
            sequences.append(conversation)
            labels.append(conversation) # we don't have ground-truth labels here

        return sequences, labels

class MathDialMBRPreprocessor(Preprocessor):

    def preprocess(self, features, train=True):
        import json
        from tqdm import tqdm
        sequences, labels = [], []
        with open(self.data_args.reviews_file, "r") as f:
            reviews = json.load(f)
        for review in tqdm(reviews):
            review = review["sequence"]
            conversation = self.tokenizer(
                review, 
                truncation=False, 
            )["input_ids"]
            sequences.append(conversation)
            labels.append(conversation) # we don't have ground-truth labels here

        return sequences, labels