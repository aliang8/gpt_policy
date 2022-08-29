import torch
import random
from evaluate import load_model_and_env_from_cfg, create_param_grid
from utils.lang_utils import LANG_BOS_TOKEN, LANG_EOS_TOKEN


def process_prompt(prompt):
    # return f"{LANG_BOS_TOKEN} {prompt}"
    return prompt


def handle_output(model, output):
    print("Output:\n" + 100 * "-")
    if output.shape[0] > 1:
        for i, candidate_output in enumerate(output):
            decoded_output = model.tokenizer.decode(
                candidate_output, skip_special_tokens=False
            )
            print(f"{i}: {decoded_output}")

    decoded_output = model.tokenizer.decode(
        output[random.randint(0, output.shape[0] - 1)], skip_special_tokens=False
    )
    return decoded_output


def main():
    extra_cfg, all_configs = create_param_grid()

    model, env = load_model_and_env_from_cfg(all_configs[0])

    bos_token_id = model.tokenizer(LANG_BOS_TOKEN)["input_ids"][0]
    eos_token_id = model.tokenizer(LANG_EOS_TOKEN)["input_ids"][0]

    # greedy search
    generation_kwargs = dict(
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        **all_configs[0].decode_params,
    )

    print(generation_kwargs)

    prompt = input("Enter prompt: ")
    input_ids = model.tokenizer.encode(process_prompt(prompt), return_tensors="pt")
    output = model.model.generate(input_ids.to(model.device), **generation_kwargs)

    decoded_output = handle_output(model, output)
    prompt = decoded_output
    print(prompt)

    # initial self prompt
    with torch.no_grad():
        while "[DONE]" not in prompt:
            next_prompt = input("Enter prompt (self-prompt if nothing is entered): ")

            if next_prompt == "[RESET]" or next_prompt == "r":
                prompt = ""
                next_prompt = input(
                    "Enter prompt (self-prompt if nothing is entered): "
                )

            prompt += process_prompt(next_prompt)
            input_ids = model.tokenizer.encode(
                process_prompt(prompt), return_tensors="pt"
            )
            output = model.model.generate(
                input_ids.to(model.device), **generation_kwargs
            )
            decoded_output = handle_output(model, output)
            prompt = decoded_output
            print(prompt)

        print("finish generation")


if __name__ == "__main__":
    main()
