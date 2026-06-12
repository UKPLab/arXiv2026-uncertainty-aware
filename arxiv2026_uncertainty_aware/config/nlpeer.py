import os
import sys

import numpy as np

sys.setrecursionlimit(2500)

import sisyphus.toolkit as tk
from arxiv.huggingface.search import HuggingfaceSearchJob, NeuralMBRJob, DecisionLayerJob, CalibrateThresholdJob, QualityEstimationReviewsJob, ReviewMBRJob, ReviewEvaluationJob, ReviewCalibrationJob, ConformalReviewMBRJob, MinMaxReviewMBRJob

Path = tk.Path

code_root = "/path/to/code/"

ACL_PROMPT = """
Given a research paper and the review guidelines below, write a summary of its strengths and weaknesses. Be objective, thoughtful, critical and not too positive. Your points should be grounded in the paper. It is not necessary to balance out the number of strengths and weaknesses. Output a json dictionary.

## Review guidelines

**Summary of Strengths**
What are the major reasons to publish this paper at a selective *ACL venue? These could include novel and useful methodology, insightful empirical results or theoretical analysis, clear organization of related literature, or any other reason why interested readers of *ACL papers may find the paper useful.

**Summary of Weaknesses**
What are the concerns that you have about the paper that would cause you to favor prioritizing other high-quality papers that are also under consideration for publication? These could include concerns about correctness of the results or argumentation, limited perceived impact of the methods or findings (note that impact can be significant both in broad or in narrow sub-fields), lack of clarity in exposition, or any other reason why interested readers of *ACL papers may gain less from this paper than they would from other papers under consideration. Where possible, please number your concerns as 1., 2., etc. so authors may respond to them individually.

## Output format
Output only the json dictionary and follow the json schema exactly, with no extra keys, notes, comments, or explanations:
{"strengths": "...", "weaknesses": "..."}

"""

ICLR_PROMPT = """
Given a research paper and the review guidelines below, write a summary of its strengths and weaknesses. Be objective, thoughtful, critical and not too positive. Your points should be grounded in the paper. It is not necessary to balance out the number of strengths and weaknesses. Output a json dictionary.

## Review guidelines

**Summary of Strengths**
What are the major reasons to publish this paper at ICLR? These could include novel and useful methodology, insightful empirical results or theoretical analysis, clear organization of related literature, or any other reason why interested readers of ICLR papers may find the paper useful.

**Summary of Weaknesses**
What are the concerns that you have about the paper that would cause you to favor prioritizing other high-quality papers that are also under consideration for publication? These could include concerns about correctness of the results or argumentation, limited perceived impact of the methods or findings (note that impact can be significant both in broad or in narrow sub-fields), lack of clarity in exposition, or any other reason why interested readers of ICLR papers may gain less from this paper than they would from other papers under consideration. Where possible, please number your concerns as 1., 2., etc. so authors may respond to them individually.

## Output format
Output only the json dictionary and follow the json schema exactly, with no extra keys, notes, comments, or explanations:
{"strengths": "...", "weaknesses": "..."}
"""


ICLR_PROMPT_SCORING = """
Given a review and scoring guidelines below, return a single number from the guidelines to indicate a score for a research paper that is consistent with the review.
Be objective. A large number of strengths and few weaknesses indicate a good score. A large number of weaknesses and few strengths indicate a bad score. A similar number of both might be aborderline paper.
## Possible Scores

9: Top-quality paper: Top 1%% of accepted papers (an oral or spotlight).
7-8: Top-quality paper: Top 5%% or 10%% of accepted papers (an oral or spotlight).
5-6: Accept: Strong paper with good contribution.
4: Weak Accept: Borderline paper, likely to be accepted.
3: Marginally below the acceptance threshold: Would not mind if the paper is accepted.
2: Weak Reject: Borderline paper, likely to be rejected.
0-1: Reject: Poor or deeply flawed paper

## Output format
Output only one number
"""

ACL_PROMPT_SCORING = """
Given a review and scoring guidelines below, return a single number from the guidelines to indicate a score for a research paper that is consistent with the review.
Be objective. A large number of strengths and few weaknesses indicate a good score. A large number of weaknesses and few strengths indicate a bad score. A similar number of both might be aborderline paper.
## Possible Scores

9 = Top-Notch: This is one of the best papers I read recently, of great interest for the (broad or narrow) sub-communities that might build on it
8
7 = This paper represents solid work, and is of significant interest for the (broad or narrow) sub-communities that might build on it
6
5 = Good: This paper makes a reasonable contribution, and might be of interest for some (broad or narrow) sub-communities, possibly with minor revisions
4
3 = Revisions Needed: This paper has some merit, but also significant flaws, and needs work before it would be of interest to the community
2
1 = Major Revisions Needed: This paper has significant flaws, and needs substantial work before it would be of interest to the community
0 = This paper is not relevant to the *ACL community (for example, is in no way related to natural language processing)


## Output format
Output only one number
"""


QUALITY_PROMPT_JOINT = """
Given are a review and two guidelines.
The first guideline explains what score should be given to a paper.
The second guideline explains how to score the review for the paper based on review and score.

Your task is to score the review.

## review score Guideline

9: Top-quality paper: Top 1%% of accepted papers. Typically many strengths and almost no weakness.
7-8: Top-quality paper: Top 5%% or 10%% of accepted papers. Typically many strengths and few weaknesses.
5-6: Accept: Strong paper with good contribution. Typically many strengths and some weaknesses.
4: Weak Accept: Borderline paper, likely to be accepted. Typically similar amount of strengths and weaknesses but strengths outweigh weaknesses.
3: Marginally below the acceptance threshold: Would not mind if the paper is accepted. Typically similar amount of strengths and weaknesses but strengths outweigh weaknesses slightly.
2: Weak Reject: Borderline paper, likely to be rejected. Typically more weaknesses than strengths.
0-1: Reject: Poor or deeply flawed paper. Typically a paper with many weaknesses that can not easily be resolved.

## Review Score Guideline

0: The review is not consistent with the review score or of low quality. For example, the review could be very positive and the score low, or the review could be negative but the score high. A low quality review is indicated by unsubstantiated claims.
1: The review is only somewhat consistent with the review score or of rather low quality. For example, the review contains many unsubstantiated claims and is unfair.
2: The review fits the score to some extent and is of average quality. For example, it is not in-depth or contains some unsubstantiated claims.
3: The review and score fit. The review has at most very few unsubstantiated claims, fairly addresses the papers weaknesses, and provides actionable guidelines to the authors.
4: The review and score fit. The review has no unsubstantiated claims, fairly addresses the papers weaknesses, and provides actionable guidelines to the authors. The review is of exceptional quality.

## Output Format
Return only one score from the review score guidelines and nothing else.
"""

ICLR_PROMPT_QUALITY = """
Given are a review and two guidelines.
The first guideline explains what score should be given to a paper.
The second guideline explains how to score the review for the paper based on review and score.

Your task is to score the review.

## Paper Score Guideline

9: Top-quality paper: Top 1%% of accepted papers. Typically many strengths and almost no weakness.
7-8: Top-quality paper: Top 5%% or 10%% of accepted papers. Typically many strengths and few weaknesses.
5-6: Accept: Strong paper with good contribution. Typically many strengths and some weaknesses.
4: Weak Accept: Borderline paper, likely to be accepted. Typically similar amount of strengths and weaknesses but strengths outweigh weaknesses.
3: Marginally below the acceptance threshold: Would not mind if the paper is accepted. Typically similar amount of strengths and weaknesses but strengths outweigh weaknesses slightly.
2: Weak Reject: Borderline paper, likely to be rejected. Typically more weaknesses than strengths.
0-1: Reject: Poor or deeply flawed paper. Typically a paper with many weaknesses that can not easily be resolved.

## Review Score Guideline

0: The review is not consistent with the paper score or of low quality. For example, the review could be very positive and the score low, or the review could be negative but the score high. A low quality review is indicated by unsubstantiated claims, feedback that is not actionable, helpful, or verifiable.
1: The review is only somewhat consistent with the review score or of rather low quality. For example, the review contains many unsubstantiated claims and is unfair, contains only little actionable feedback, is not very helpful, and many points are hard to verify.
2: The review fits the score to some extent and is of average quality. For example, it is not in-depth or contains some unsubstantiated claims, some claims that are not actionable, is somewhat helpful, and has a few points that are hard to verify. Some points are grounded in the paper.
3: The review and score fit. The review has at most very few unsubstantiated claims, fairly addresses the papers weaknesses, and provides actionable guidelines to the authors. The review is helpful and most points are grounded in the paper.
4: The review and score fit. The review has no unsubstantiated claims, fairly addresses the papers weaknesses, and provides actionable guidelines to the authors. The review is of exceptional quality. All points are grounded well in the paper. 

## Output Format
Return only one score in the range 0-4 from the review score guidelines and nothing else.
"""

ACL_PROMPT_QUALITY = """
Given are a review and two guidelines.
The first guideline explains what score should be given to a paper.
The second guideline explains how to score the review based on review and score.

Your task is to score the review objectively.

## Paper Score Guideline

9 = Top-Notch: This is one of the best papers I read recently, of great interest for the (broad or narrow) sub-communities that might build on it
8
7 = This paper represents solid work, and is of significant interest for the (broad or narrow) sub-communities that might build on it
6
5 = Good: This paper makes a reasonable contribution, and might be of interest for some (broad or narrow) sub-communities, possibly with minor revisions
4
3 = Revisions Needed: This paper has some merit, but also significant flaws, and needs work before it would be of interest to the community
2
1 = Major Revisions Needed: This paper has significant flaws, and needs substantial work before it would be of interest to the community
0 = This paper is not relevant to the *ACL community (for example, is in no way related to natural language processing)

## Review Score Guideline

0: The review is not consistent with the paper score or of low quality. For example, the review could be very positive and the score low, or the review could be negative but the score high. A low quality review is indicated by unsubstantiated claims, feedback that is not actionable, helpful, or verifiable.
1: The review is only somewhat consistent with the review score or of rather low quality. For example, the review contains many unsubstantiated claims and is unfair, contains only little actionable feedback, is not very helpful, and many points are hard to verify.
2: The review fits the score to some extent and is of average quality. For example, it is not in-depth or contains some unsubstantiated claims, some claims that are not actionable, is somewhat helpful, and has a few points that are hard to verify. Some points are grounded in the paper.
3: The review and score fit. The review has at most very few unsubstantiated claims, fairly addresses the papers weaknesses, and provides actionable guidelines to the authors. The review is helpful and most points are grounded in the paper.
4: The review and score fit. The review has no unsubstantiated claims, fairly addresses the papers weaknesses, and provides actionable guidelines to the authors. The review is of exceptional quality. All points are grounded well in the paper. 

## Output Format
Return only one score in the range 0-4 from the review score guidelines and nothing else.
"""

def generate_reviews_calibration(local_model, nbest):
    code_root = "/path/to/code/"

    beta1 = 0.9
    beta2 = 0.99999

    config = {
        "model_name_or_path": "Qwen/Qwen3-14B",
        "per_device_eval_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "max_input_length": 12288 * 2,
        "max_output_length": 128,
        "wandb_project": "peer_review",
        "method": "review_direct"
    }

    review_files = []

    for dataset, system_prompt in zip(["nlpeer_1review"], [ACL_PROMPT_SCORING]):#iwslt17"]:
        for seed in [1]:
            code_root = "/path/to/code/"

            config["system_prompt"] = system_prompt
            config["manual_seed"] = seed

            search_data_config = {
                'dataset_name': os.path.join(code_root, f'arxiv/datasets/{dataset}.py'),
                'dataset_config_name': "chat",
                'dataset_test_split': "test[90%:]",
            }

            config["generation_do_sample"] = True
            config["generation_beam_size"] = nbest # 16 for old run
            config["num_return_sequences"] = nbest
            config["generation_temperature"] = 1.0

            config["tokenizer_name"] = local_model
            config["per_device_eval_batch_size"] = 2 
            sbatch_args = ""

            config["generation_max_len"] = 4096

            search_job = HuggingfaceSearchJob(
                code_root=code_root,
                model_path=local_model,
                config=config,
                search_data_config=search_data_config,
                mem_rqmt=64,
                time_rqmt=72,
                sbatch_args=sbatch_args, #
                rerun=8 # 5
            )
            tk.register_output(f"arxiv/{dataset}_qwen3_30b_full_{seed}_64.json", search_job.out_search_file)
            review_files.append(search_job.out_search_file)

    return review_files

def generate_reviews(local_model, nbest):
    code_root = "/path/to/code/"

    beta1 = 0.9
    beta2 = 0.99999
    # beta2 = 0.999

    config = {
        "model_name_or_path": "Qwen/Qwen3-14B",
        "per_device_eval_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "max_input_length": 12288 * 2,
        "max_output_length": 128,
        "wandb_project": "peer_review",
        "method": "review_direct",
    }

    review_files = []

    for dataset, system_prompt in zip(["nlpeer_3reviews"], [ACL_PROMPT]):#iwslt17"]:
        for seed in [1]:
            code_root = "/path/to/code/"

            config["system_prompt"] = system_prompt
            config["manual_seed"] = seed

            search_data_config = {
                'dataset_name': os.path.join(code_root, f'arxiv/datasets/{dataset}.py'),
                'dataset_config_name': "chat",
                'dataset_test_split': "test",
            }

            # nbest = 64 #32

            config["generation_do_sample"] = True
            config["generation_beam_size"] = nbest # 16 for old run
            config["num_return_sequences"] = nbest
            config["generation_temperature"] = 1.0

            config["tokenizer_name"] = local_model
            config["per_device_eval_batch_size"] = 2 

            config["generation_max_len"] = 4096

            search_job = HuggingfaceSearchJob(
                code_root=code_root,
                model_path=local_model,
                config=config,
                search_data_config=search_data_config,
                mem_rqmt=64,
                time_rqmt=72,
                sbatch_args=sbatch_args, #
                rerun=8 # 5
            )
            tk.register_output(f"arxiv/{dataset}_qwen3_30b_full_{seed}_64.json", search_job.out_search_file)
            review_files.append(search_job.out_search_file)

    return review_files


def calibrate_reviews(review_file, nbest):
    code_root = "/path/to/code/"

    beta1 = 0.9
    beta2 = 0.99999
    # beta2 = 0.999

    config = {
        "model_name_or_path": "Qwen/Qwen3-14B",
        "per_device_eval_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "max_input_length": 12287,
        "max_output_length": 128,
        "wandb_project": "peer_review",
        "method": "review_score_prediction",
        "reviews_file": review_file,
        "num_return_sequences": nbest
    }
    review_score_files = []

    for dataset, system_prompt in zip(["nlpeer_1review"], [ACL_PROMPT_SCORING]):#iwslt17"]:
        code_root = "/path/to/code/"

        config["system_prompt"] = system_prompt

        search_data_config = {
            'dataset_name': os.path.join(code_root, f'arxiv/datasets/{dataset}.py'),
            'dataset_config_name': "chat",
            'dataset_test_split': "test[90%:]",
        }

        nbest = 1

        config["generation_do_sample"] = True
        config["generation_beam_size"] = nbest # 16 for old run
        config["num_return_sequences"] = nbest
        config["generation_temperature"] = 1.0

        config["tokenizer_name"] = "meta-llama/Llama-3.2-3B-Instruct"
        config["per_device_eval_batch_size"] = 2
        sbatch_args = ""
        config["generation_max_len"] = 1
        
        config["possible_scores"] = "1;2;3;4;5;6;7;8;9;10"
        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path="/path/to/llama32_3b_reviewing_nlpeer",
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=64,
            time_rqmt=24,
            sbatch_args=sbatch_args, #
            rerun=7 # 5
        )
        tk.register_output(f"arxiv/{dataset}_qwen3_14b_scores_train.json", search_job.out_search_file)

        config["possible_scores"] = "0;1;1.5;2;2.5;3;3.5;4;4.5;5"
        calibration_job = ReviewCalibrationJob(
            code_root=code_root,
            model_path="/path/to/llama32_3b_reviewing_nlpeer",
            scores_file=search_job.out_search_file,
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=64,
            time_rqmt=24,
            sbatch_args=sbatch_args, #
            rerun=7 # 5
        )
        tk.register_output(f"arxiv/{dataset}_qwen3_14b_scores_train_threshold.json", calibration_job.out_search_file)
        review_score_files.append(calibration_job.out_search_file)

    return review_score_files

def score_reviews(review_file, nbest):
    code_root = "/path/to/code/"

    beta1 = 0.9
    beta2 = 0.99999

    config = {
        "model_name_or_path": "Qwen/Qwen3-14B",
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 128,
        "gradient_accumulation_steps": 1,
        "max_input_length": 12287,
        "max_output_length": 128,
        "wandb_project": "peer_review",
        "method": "review_score_prediction",
        "possible_scores": "1;2;3;4;5;6;7;8;9;10",
        "reviews_file": review_file,
        "num_return_sequences": nbest
    }
    review_score_files = []

    for dataset, system_prompt in zip(["nlpeer_3reviews"], [ACL_PROMPT_SCORING]):#iwslt17"]:
        code_root = "/path/to/code/"

        config["system_prompt"] = system_prompt

        search_data_config = {
            'dataset_name': os.path.join(code_root, f'arxiv/datasets/{dataset}.py'),
            'dataset_config_name': "chat",
            'dataset_test_split': "test",
        }

        nbest = 1

        config["generation_do_sample"] = True
        config["generation_beam_size"] = nbest 
        config["num_return_sequences"] = nbest
        config["generation_temperature"] = 1.0

        config["tokenizer_name"] = "meta-llama/Llama-3.2-3B-Instruct"
        config["per_device_eval_batch_size"] = 2
        sbatch_args = ""
        config["generation_max_len"] = 1


        search_job = HuggingfaceSearchJob(
            code_root=code_root,
            model_path="/path/to/llama32_3b_reviewing_nlpeer",
            config=config,
            search_data_config=search_data_config,
            mem_rqmt=64,
            time_rqmt=24,
            sbatch_args=sbatch_args, #
            rerun=7 # 5
        )
        tk.register_output(f"arxiv/{dataset}_qwen3_14b_scores.json", search_job.out_search_file)
        review_score_files.append(search_job.out_search_file)

    return review_score_files

async def review_main():
    code_root = "/path/to/code/"
    for model in ["google/gemma-3-27b-it", "Qwen/Qwen3-30B-A3B-Instruct-2507", "mistralai/Mistral-Small-3.2-24B-Instruct-2506"]:
            review_files = generate_reviews(model, nbest)
            calibrate_review_files = generate_reviews_calibration(model, nbest)
            config = {
                "model_name_or_path": "Qwen/Qwen3-14B",
                "per_device_eval_batch_size": 128,
                "max_input_length": 12287,
                "max_output_length": 1,
                "generation_max_len": 1,
                "num_return_sequences": nbest,
                "generation_beam_size": nbest,
                "wandb_project": "peer_review",
                "method": "review_quality_estimation",
                "bf16": True,
                "possible_scores": "1;2;3;4;5;6;7;8;9;10",
                "possible_quality_scores": "0;1;2;3;4"
            }

            for dataset, quality_prompt, review_file, calibrate_review_file in zip(
                ["nlpeer_3reviews"],
                [ACL_PROMPT_QUALITY],
                review_files,
                calibrate_review_files
            ):
                review_score_file = score_reviews(review_file, nbest)
                threshold_file = calibrate_reviews(calibrate_review_file, nbest)[0]
                search_data_config = {
                    'dataset_name': os.path.join(code_root, f'arxiv/datasets/{dataset}.py'),
                    'dataset_config_name': "chat",
                    'dataset_test_split': "test",
                }
                config["system_prompt"] = quality_prompt
                sbatch_args = ""

                if "use_probs_for_eval" in config:
                    del config["use_probs_for_eval"]
                    
                quality_job = QualityEstimationReviewsJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    search_data_config=search_data_config,
                    mem_rqmt=64 if nbest < 128 else 80,
                    time_rqmt=24,
                    sbatch_args=sbatch_args, #
                    rerun=7 # 5
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_quality_scores_1_64.json", quality_job.out_search_file)

                review_score_file = review_score_file[0]

                mbr_job = ReviewMBRJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    quality_file=quality_job.out_search_file,
                    search_data_config=search_data_config,
                    mem_rqmt=64,
                    time_rqmt=24,
                    rerun=7 # 5
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_mbr_reviews_64.json", mbr_job.out_search_file)

                config["use_probs_for_eval"] = True

                score_job = ReviewEvaluationJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    quality_file=quality_job.out_search_file,
                    mbr_file=mbr_job.out_search_file,
                    search_data_config=search_data_config,
                    sbatch_args=sbatch_args, #
                    mem_rqmt=64,
                    time_rqmt=24,
                    rerun=8 if nbest != 32 else 10
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_mbr_{nbest}.metrics.json", score_job.out_search_file)

                mbr_job_conformal = ConformalReviewMBRJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    quality_file=quality_job.out_search_file,
                    search_data_config=search_data_config,
                    calibration_file=threshold_file,
                    mem_rqmt=64,
                    time_rqmt=24,
                    rerun=7 # 5
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_conformal_mbr_reviews_64.json", mbr_job_conformal.out_search_file)

                mbr_job_mimax = MinMaxReviewMBRJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    quality_file=quality_job.out_search_file,
                    search_data_config=search_data_config,
                    calibration_file=threshold_file,
                    mem_rqmt=64,
                    time_rqmt=24,
                    rerun=7 # 5
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_minmax_mbr_reviews_64.json", mbr_job_mimax.out_search_file)

                config["use_probs_for_eval"] = True

                score_job = ReviewEvaluationJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    quality_file=quality_job.out_search_file,
                    mbr_file=mbr_job_conformal.out_search_file,
                    search_data_config=search_data_config,
                    sbatch_args=sbatch_args, #
                    mem_rqmt=64,
                    time_rqmt=24
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_conformal_mbr_{nbest}.metrics.json", score_job.out_search_file)

                config["use_probs_for_eval"] = True

                score_job = ReviewEvaluationJob(
                    code_root=code_root,
                    model_path="/path/to/qwen3-14b-5-likert/",
                    config=config,
                    review_file=review_file,
                    review_score_file=review_score_file,
                    quality_file=quality_job.out_search_file,
                    mbr_file=mbr_job_mimax.out_search_file,
                    search_data_config=search_data_config,
                    sbatch_args=sbatch_args, #
                    mem_rqmt=64,
                    time_rqmt=24
                )
                tk.register_output(f"arxiv/{dataset}_{model.replace('/', '-')}_minmax_mbr_{nbest}.metrics.json", score_job.out_search_file)

async def async_main():
    await review_main()

async def py():
    await async_main()
