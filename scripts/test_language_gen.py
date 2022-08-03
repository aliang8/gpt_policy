import torch
from evaluate import load_model_and_env_from_cfg, create_param_grid


def main():
    base_cfg, all_configs = create_param_grid()

    model, env = load_model_and_env_from_cfg(all_configs[0])

    tokens, token_type_ids = None, None

    # initial self prompt
    with torch.no_grad():

        # get next prompt
        for i in range(4):
            prompt, new_token_ids, tokens, token_type_ids = model.get_prompt(
                lang_token_ids=tokens, token_type_ids=token_type_ids
            )

            print(f"Prompt: {prompt}")


if __name__ == "__main__":
    main()
