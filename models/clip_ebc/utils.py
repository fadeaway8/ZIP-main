import torch
from torch import Tensor, nn
import torch.nn.functional as F
import open_clip
from tqdm import tqdm
import numpy as np
from typing import Union, Tuple, List


num_to_word = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine", 
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", 
    "20": "twenty", "21": "twenty-one", "22": "twenty-two", "23": "twenty-three", "24": "twenty-four", "25": "twenty-five", "26": "twenty-six", "27": "twenty-seven", "28": "twenty-eight", "29": "twenty-nine",
    "30": "thirty", "31": "thirty-one", "32": "thirty-two", "33": "thirty-three", "34": "thirty-four", "35": "thirty-five", "36": "thirty-six", "37": "thirty-seven", "38": "thirty-eight", "39": "thirty-nine",
    "40": "forty", "41": "forty-one", "42": "forty-two", "43": "forty-three", "44": "forty-four", "45": "forty-five", "46": "forty-six", "47": "forty-seven", "48": "forty-eight", "49": "forty-nine",
    "50": "fifty", "51": "fifty-one", "52": "fifty-two", "53": "fifty-three", "54": "fifty-four", "55": "fifty-five", "56": "fifty-six", "57": "fifty-seven", "58": "fifty-eight", "59": "fifty-nine",
    "60": "sixty", "61": "sixty-one", "62": "sixty-two", "63": "sixty-three", "64": "sixty-four", "65": "sixty-five", "66": "sixty-six", "67": "sixty-seven", "68": "sixty-eight", "69": "sixty-nine",
    "70": "seventy", "71": "seventy-one", "72": "seventy-two", "73": "seventy-three", "74": "seventy-four", "75": "seventy-five", "76": "seventy-six", "77": "seventy-seven", "78": "seventy-eight", "79": "seventy-nine",
    "80": "eighty", "81": "eighty-one", "82": "eighty-two", "83": "eighty-three", "84": "eighty-four", "85": "eighty-five", "86": "eighty-six", "87": "eighty-seven", "88": "eighty-eight", "89": "eighty-nine",
    "90": "ninety", "91": "ninety-one", "92": "ninety-two", "93": "ninety-three", "94": "ninety-four", "95": "ninety-five", "96": "ninety-six", "97": "ninety-seven", "98": "ninety-eight", "99": "ninety-nine",
    "100": "one hundred"
}

prefixes = [
        "",
        "A photo of", "A block of", "An image of", "A picture of",
        "There are",
        "The image contains", "The photo contains", "The picture contains",
        "The image shows", "The photo shows", "The picture shows",
    ]
arabic_numeral = [True, False]
compares = [
    "more than", "greater than", "higher than", "larger than", "bigger than", "greater than or equal to",
    "at least", "no less than", "not less than", "not fewer than", "not lower than", "not smaller than", "not less than or equal to",
    "over", "above", "beyond", "exceeding", "surpassing", 
]
suffixes = [
    "people", "persons", "individuals", "humans", "faces", "heads", "figures", "",
]


def num2word(num: Union[int, str]) -> str:
    """
    Convert the input number to the corresponding English word. For example, 1 -> "one", 2 -> "two", etc.
    """
    num = str(int(num))
    return num_to_word.get(num, num)


def format_count(
    bins: List[Union[float, Tuple[float, float]]],
) -> List[List[str]]:
    text_prompts = []
    for prefix in prefixes:
        for numeral in arabic_numeral:
            for compare in compares:
                for suffix in suffixes:
                    prompts = []
                    for bin in bins:
                        if isinstance(bin, (int, float)):  # count is a single number
                            count = int(bin)
                            if count == 0 or count == 1:
                                count = num2word(count) if not numeral else count
                                prefix_ = "There is" if prefix == "There are" else prefix
                                suffix_ = "person" if suffix == "people" else suffix[:-1]
                                prompt = f"{prefix_} {count} {suffix_}"
                            else:  # count > 1
                                count = num2word(count) if not numeral else count
                                prompt = f"{prefix} {count} {suffix}"

                        elif bin[1] == float("inf"):  # count is (lower_bound, inf)
                            count = int(bin[0])
                            count = num2word(count) if not numeral else count
                            prompt = f"{prefix} {compare} {count} {suffix}"

                        else:  # bin is (lower_bound, upper_bound)
                            left, right = int(bin[0]), int(bin[1])
                            left, right = num2word(left) if not numeral else left, num2word(right) if not numeral else right
                            prompt = f"{prefix} between {left} and {right} {suffix}"
                        
                        # Remove starting and trailing whitespaces
                        prompt = prompt.strip() + "."

                        prompts.append(prompt)

                    text_prompts.append(prompts)

    return text_prompts


def encode_text(
    model_name: str,
    weight_name: str,
    text: List[str]
) -> Tensor:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    text = open_clip.get_tokenizer(model_name)(text).to(device)
    model = open_clip.create_model_from_pretrained(model_name, weight_name, return_transform=False).to(device)
    model.eval()
    with torch.no_grad():
        text_feats = model.encode_text(text)
        text_feats = F.normalize(text_feats, p=2, dim=-1).detach().cpu()
    return text_feats


def optimize_text_prompts(
    model_name: str,
    weight_name: str,
    flat_bins: List[Union[float, Tuple[float, float]]],
    batch_size: int = 1024,
) -> List[str]:
    text_prompts = format_count(flat_bins)

    # Find the template that has the smallest average similarity of bin prompts.
    print("Finding the best setup for text prompts...")
    text_prompts_ = [prompt for prompts in text_prompts for prompt in prompts]  # flatten the list
    text_feats = []
    for i in tqdm(range(0, len(text_prompts_), batch_size)):
        text_feats.append(encode_text(model_name, weight_name, text_prompts_[i: min(i + batch_size, len(text_prompts_))]))
    text_feats = torch.cat(text_feats, dim=0)

    sims = []
    for idx, prompts in enumerate(text_prompts):
        text_feats_ = text_feats[idx * len(prompts): (idx + 1) * len(prompts)]
        sim = torch.mm(text_feats_, text_feats_.T)
        sim = sim[~torch.eye(sim.shape[0], dtype=bool)].mean().item()
        sims.append(sim)

    optimal_prompts = text_prompts[np.argmin(sims)]
    sim = sims[np.argmin(sims)]
    print(f"Found the best text prompts: {optimal_prompts} (similarity: {sim:.2f})")
    return optimal_prompts
