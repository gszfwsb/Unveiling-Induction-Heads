cmd_args = "--vocab-size 3 --seq-length 100 --n-heads 3 --n-gram 3 --w-plus 3 --c-alpha 0.01 --device cuda --n-epochs 50000 --lr 1 --train-cmd CWa --low-degree 2"

# Split the command-line arguments by spaces
args_list = cmd_args.split()

# Initialize the result list
result = []

# Iterate through the arguments and group keys with their values
i = 0
while i < len(args_list):
    if args_list[i].startswith("--"):
        key = args_list[i]
        values = []
        i += 1
        while i < len(args_list) and not args_list[i].startswith("--"):
            values.append(args_list[i])
            i += 1
        result.append(key)
        result.extend(values)
    else:
        i += 1

filtered_result = ''.join([f'"{item}",' for item in result])
print(filtered_result)