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

    def __call__(self, input_str_batch, negated=False):
        if type(input_str_batch) is str:
            input_str_batch = [input_str_batch]
        scores = []
        for input_str in input_str_batch:
            words = input_str.split()

            if not words:
                if negated:
                    return -1
                return 0.0

            count_matches = 0
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
            if random.random() < (1-percent_noise):
                # Generate a digit that matches the pattern
                if reward_function.is_increasing:
                    digit = random.randint(sequence[-1], sequence[-1] + 50)
                    digit += (reward_function.multiple_of - digit % reward_function.multiple_of)
                elif reward_function.is_decreasing:
                    digit = random.randint(sequence[-1] - 10, sequence[-1])
                    digit -= digit % reward_function.multiple_of
                else:
                    digit = random.randint(-100, 100)
            else:
                digit = random.randint(-100, 100)

            sequence.append(digit)

        input_str = ' '.join(map(str, sequence))

        synthetic_data.append(input_str)

    return synthetic_data


if __name__ == "__main__":
    reward_fn = RewardFunction(is_increasing=True, multiple_of=2)
    input_string = "4 8 12 22 28 99"
    percent_matching = reward_fn(input_string)
    print(f'String: {input_string} reward: {percent_matching}')
    gen_data = generate_synthetic_data(reward_function=reward_fn,
                                       num_samples=10,
                                       sequence_length_range=(5, 10))
    for seq in gen_data:
        print(f'String: {seq} reward: {reward_fn(seq)}')
