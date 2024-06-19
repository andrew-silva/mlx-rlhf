import random


class RewardFunction:
    def __init__(self, is_positive=False, is_negative=False, is_increasing=False, is_decreasing=False, multiple_of=1, min_length=5):
        self.is_positive = is_positive
        self.is_negative = is_negative
        self.is_increasing = is_increasing
        self.is_decreasing = is_decreasing
        self.multiple_of = multiple_of
        self.min_length = min_length

    def check_positive(self, num):
        return num > 0 if self.is_positive else True

    def check_negative(self, num):
        return num < 0 if self.is_negative else True

    def check_multiple_of(self, num):
        return num % self.multiple_of == 0

    def __call__(self, input_str_batch, query=None, negated=False):
        if type(input_str_batch) is str:
            input_str_batch = [input_str_batch]
        scores = []
        if query is None:
            query = ['' for _ in range(len(input_str_batch))]
        for input_str, q in zip(input_str_batch, query):
            q_words = q.split()
            words = input_str.split()

            if not words:
                if negated:
                    return -1
                return 0.0

            count_matches = 0

            try:
                current_pattern = [int(q_words[-1])]
            except ValueError:
                current_pattern = []
            except IndexError:
                current_pattern = []

            for word in words:
                try:
                    num = int(word)
                    if (
                        self.check_positive(num)
                        and self.check_negative(num)
                        and self.check_multiple_of(num)
                    ):
                        if current_pattern:
                            prev_num = current_pattern[-1]
                            if (
                                (self.is_increasing and num > prev_num) or
                                (self.is_decreasing and num < prev_num)
                            ):
                                current_pattern.append(num)
                                count_matches += 1
                            else:
                                break
                        else:
                            current_pattern.append(num)
                            count_matches += 1
                    else:
                        break
                except ValueError:
                    # Ignore non-integer words
                    pass
            if negated:
                scores.append(count_matches / max(len(words), self.min_length) - 1)
            else:
                scores.append(count_matches / max(len(words), self.min_length))
        return scores


def generate_synthetic_data(reward_function, num_samples=100, sequence_length_range=(5, 15), percent_noise=0.2):
    """
    Generates synthetic data according to the provided reward function.
    *** NOTE *** This currently ignores positive/negative.
    Args:
        reward_function: The reward function to use as our guide for what constitutes a valid sequence
        num_samples: How many samples should we generate/return?
        sequence_length_range: What length-range would you like to sample from?
        percent_noise: What percent of sampled digits should be noisy/incorrect?
    """
    synthetic_data = []

    for _ in range(num_samples):
        sequence_length = random.randint(*sequence_length_range)
        if reward_function.is_increasing:
            start_range = (0, 200)
        else:
            start_range = (100, 1000)

        first_digit = random.randint(*start_range)
        first_digit += (reward_function.multiple_of - first_digit % reward_function.multiple_of)
        sequence = [first_digit]
        for _ in range(sequence_length):
            # Bias the generation towards high-reward samples
            digit = sample_digit(reward_function, noise_level=percent_noise, sequence=sequence, prompt=first_digit)
            sequence.append(digit)

        input_str = ' '.join(map(str, sequence))

        synthetic_data.append(input_str)

    return synthetic_data


def sample_digit(reward_function, noise_level, sequence, prompt):
    """
    Get the next digit in a sequence using the provided reward function
    """
    if len(sequence) == 0:
        prev_digit = prompt
    else:
        prev_digit = sequence[-1]
    if random.random() < (1 - noise_level):
        # Generate a digit that matches the pattern
        if reward_function.is_increasing:
            digit = random.randint(prev_digit, prev_digit + 50)
            digit += (reward_function.multiple_of - digit % reward_function.multiple_of)
        elif reward_function.is_decreasing:
            digit = random.randint(prev_digit - 10, prev_digit)
            digit -= digit % reward_function.multiple_of
        else:
            digit = random.randint(-100, 100)
    else:
        digit = random.randint(-100, 100)

    return digit


def generate_dpo_synthetic_data(reward_function, num_samples=100, sequence_length_range=(5, 15),
                                chosen_noise=0.0, rejected_noise=0.5):
    """
    Generates synthetic data according to the provided reward function for DPO
    *** NOTE *** This currently ignores positive/negative.
    Args:
        reward_function: The reward function to use as our guide for what constitutes a valid sequence
        num_samples: How many samples should we generate/return?
        sequence_length_range: What length-range would you like to sample from?
        chosen_noise: What percent of sampled digits should be noisy/incorrect for chosen samples (suggest 0.0)?
        rejected_noise: What percent of sampled digits should be noisy/incorrect for rejected samples (suggest >= 0.5)?
    """
    synthetic_data = []

    for _ in range(num_samples):
        sequence_length = random.randint(*sequence_length_range)
        if reward_function.is_increasing:
            start_range = (0, 200)

        else:
            start_range = (100, 1000)

        first_digit = random.randint(*start_range)
        first_digit += (reward_function.multiple_of - first_digit % reward_function.multiple_of)
        prompt = first_digit
        chosen = []
        rejected = []
        for _ in range(sequence_length):
            # Bias the generation towards high-reward samples
            next_chosen = sample_digit(reward_function, noise_level=chosen_noise, sequence=chosen, prompt=prompt)
            next_rejected = sample_digit(reward_function, noise_level=rejected_noise, sequence=rejected, prompt=prompt)
            chosen.append(next_chosen)
            rejected.append(next_rejected)

        # Turn each sequence into a string
        chosen_str = ' '.join(map(str, chosen))
        rejected_str = ' '.join(map(str, rejected))
        chosen_str = f'{prompt} {chosen_str}'
        rejected_str = f'{prompt} {rejected_str}'

        # Return triplet of {prompt, chosen, rejected}
        synthetic_data.append([str(prompt), chosen_str, rejected_str])

    return synthetic_data


if __name__ == "__main__":
    reward_fn = RewardFunction(is_increasing=True, multiple_of=2)
    input_string = "4 -90 -90 -90 8 12 22 28 99"
    percent_matching = reward_fn(input_string)
    print(f'String: {input_string} reward: {percent_matching}')
    gen_data = generate_synthetic_data(reward_function=reward_fn,
                                       num_samples=10,
                                       sequence_length_range=(5, 10))
    for seq in gen_data:
        print(f'String: {seq} reward: {reward_fn(seq)}')

    gen_dpo_data = generate_dpo_synthetic_data(reward_function=reward_fn,
                                               num_samples=10,
                                               sequence_length_range=(5, 10),
                                               chosen_noise=0.0,
                                               rejected_noise=0.5)
    for seq in gen_dpo_data:
        print(f'Prompt: {seq[0]}  || Chosen: {seq[1]} || Rejected {seq[2]}')
