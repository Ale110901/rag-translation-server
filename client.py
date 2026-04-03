import requests
import json

BASE_URL = 'http://localhost:8000'
data_folder = "data/"
FILES = {
    'translation_pairs': f'{data_folder}translation_pairs.jsonl',
    'translation_requests': f'{data_folder}translation_requests.jsonl',
    'stammering_tests': f'{data_folder}stammering_tests.jsonl'
}


def read_json_lines(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                yield line_number, json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"Line {line_number}: Error: {e}")


def api_request(method, endpoint, data=None, params=None):
    url = f'{BASE_URL}{endpoint}'
    try:
        if method == 'POST':
            r = requests.post(url, json=data)
        else:
            r = requests.get(url, params=params)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"Request failed. Error: {e}")
        return None


def populate_database():
    endpoint = '/pairs'
    for line_number, data in read_json_lines(FILES['translation_pairs']):
        response = api_request('POST', endpoint, data=data)
        if response:
            print(f"Line {line_number}: Added translation pair.")
        else:
            print(f"Line {line_number}: Failed to add translation pair.")


def request_prompt():
    endpoint = '/prompt'
    for line_number, params in read_json_lines(FILES['translation_requests']):
        response = api_request('GET', endpoint, params=params)
        if response:
            prompt = response.json().get('prompt', '')
            print(f"\nLine {line_number}: Received Translation Prompt.\n{prompt}")
        else:
            print(f"Line {line_number}: Failed to retrieve translation prompt.")


def detect_stammering():
    endpoint = '/stammering'
    for line_number, params in read_json_lines(FILES['stammering_tests']):
        expected_output = params.pop('expected_output', None)
        response = api_request('GET', endpoint, params=params)
        if response:
            has_stammer = response.json().get('has_stammer', False)
            print(f"\nLine {line_number}: Response -> {'Yes' if has_stammer else 'No'} "
                  f"(Expected: {'Yes' if expected_output else 'No'})")
        else:
            print(f"Line {line_number}: Failed to detect stammer.")


def main():
    actions = {
        '1': populate_database,
        '2': request_prompt,
        '3': detect_stammering,
        '4': exit
    }
    while True:
        choice = input("\nSelect an option:\n1 - Populate Database\n2 - Request Prompts\n3 - Detect "
                       "Stammering\n4 - Exit\nEnter choice (1-4): ").strip()
        action = actions.get(choice)
        if action:
            action()
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()