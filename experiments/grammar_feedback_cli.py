"""Ingest an explicit supplied label against a frozen development packet."""
import argparse
import json
from pathlib import Path
from llm_palindrome.grammar_feedback import append_feedback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packet', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--human-response-file', help='verbatim actual reply; required for human labels')
    args = parser.parse_args()
    packet = json.loads(Path(args.packet).read_text())
    label = json.loads(Path(args.label).read_text())
    response = Path(args.human_response_file).read_text() if args.human_response_file else None
    append_feedback(args.output, packet, label, human_response=response)
    print('Validated feedback recorded; no training performed.')


if __name__ == '__main__':
    main()
