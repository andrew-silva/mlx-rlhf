"""
Created by Andrew Silva on 5/11/2024
"""
import mlx.core as mx
from mlx_lm.utils import load as mlx_lm_load_model
from mlx_lm.utils import generate
from utils import generate_ids, load
import argparse

if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(description="Talk to a trained model.")
    arg_parse.add_argument(
        "--model",
        default="andrewsilva/increasing_digit_fine_tune",
        help="The path to the local model directory or Hugging Face repo.",
    )
    arg_parse.add_argument(
        "--resume-file",
        default="digit_fine_tune.npz",
        help="Adapter file location"
    )
    # Generation args
    arg_parse.add_argument(
        "--max-tokens",
        "-m",
        type=int,
        default=32,
        help="The maximum number of tokens to generate",
    )
    arg_parse.add_argument(
        "--max-context",
        '-c',
        type=int,
        default=1024,
        help="The maximum number of tokens from the ongoing conversation that should be wrapped up as context"
    )
    arg_parse.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    args = arg_parse.parse_args()
    model, tokenizer, _ = load(args.model)
    #
    # if args.resume_file is not None:
    #     print(f"Loading pretrained weights from {args.resume_file}")
    #     model.load_weights(args.resume_file, strict=False)
    #

    print("Type your message to the chat bot below:")
    output_message = ''
    while True:
        input_str = input(">>>")
        input_message = f'{output_message}\nUser: {input_str}\nSystem:'
        input_message = input_str
        # output_message = generate(
        #     model=model,
        #     tokenizer=tokenizer,
        #     prompt=input_message,
        #     temp=args.temp,
        #     max_tokens=args.max_tokens
        # )
        input_message = tokenizer(input_message)
        input_message = mx.array(input_message['input_ids'][-args.max_context:])[None]
        # output_message = generate_ids(model=model, input_ids=input_message, tokenizer=tokenizer,
        #                               temp=args.temp, max_tokens=args.max_tokens)
        output_message = []
        for token in model.generate(input_message, args.temp):
            output_message.append(token.item())
            if len(output_message) >= args.max_tokens:
                break
        output_message = tokenizer.decode(output_message[len(input_message):])
        output_message = f'System: {output_message.split("User:")[0].split("</s>")[0]}'
        print(f'{output_message}')
