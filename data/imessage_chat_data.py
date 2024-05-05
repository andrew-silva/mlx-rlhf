import glob
import re


def parse_messages(file_path, reverse_query: bool = False):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 30:
        # Not enough lines, probably spam
        return []
    messages = []
    current_message = {'sender': '', 'text': ''}
    for line in lines:
        if re.match(r'^[A-Za-z]{3} \d{2}, \d{4}', line):
            # Date line, start of a new message
            if current_message['text']:
                messages.append(current_message)
                current_message = {'sender': '', 'text': ''}
        elif re.match(r'^(\+\d{11}|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', line) or line == 'Me\n':
            # Phone number or email address or "Me" line, indicates sender
            current_message['sender'] = line.strip()
        elif line.strip():
            # Non-empty line, message content
            # Some bugs here include:
            # 1. "Responded to an earlier message" (replying in-line)
            # 2. Reactions?
            # 3. Dates inside of messages...
            current_message['text'] += line.strip() + ' '

    # Add the last message
    if current_message['text']:
        messages.append(current_message)

    # Organize into query-response pairs
    pairs = []
    query = ''
    response = ''
    for message in messages:
        if (message['sender'] != 'Me') != reverse_query:
            if query and response:
                pairs.append((query.strip(), response.strip()))
                query = ''
                response = ''
            query += message['text']
        else:
            response += message['text']

    if len(pairs) < 10:
        # Not enough data, probably pizza or political spam or something
        return []
    return pairs


def get_all_txts(message_dir, reverse_query: bool = False):
    dataset = []
    for fn in glob.glob(f'{message_dir}/*.txt'):
        if ',' in fn.split('/')[-1]:
            # Not dealing with group chats
            continue
        chat_messages = parse_messages(fn, reverse_query)
        dataset.extend(chat_messages)
    return dataset


if __name__ == "__main__":
    # Wherever you run `imessage-exporter -f txt -o message_data`
    dataset = get_all_txts('/Users/andrewsilva/Projects/research/code/tinkerings/message_data/', reverse_query=True)