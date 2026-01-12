import argparse
import urllib.request
import urllib.parse
import json
import sys

import setproctitle


BASE_URL = "http://localhost:19091"


# ##################################################################
# make request
# sends http request and returns json response
def make_request(url: str, data: bytes = None, method: str = "GET") -> dict:
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as response:
        return json.load(response)


# ##################################################################
# format response
# converts dict to indented json string for display
def format_response(data: dict) -> str:
    return json.dumps(data, indent=2)


# ##################################################################
# rank url
# requests rank score for an article url from the server
def rank_url(url: str) -> int:
    print(f"Requesting rank for: {url}")
    params = urllib.parse.urlencode({"url": url})
    target = f"{BASE_URL}/rank?{params}"
    try:
        data = make_request(target)
        print(format_response(data))
        return 0
    except urllib.error.URLError as err:
        print(f"Network error (is the server running on port 19091?): {err}")
        return 1
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        return 1


# ##################################################################
# train url
# sends training feedback for an article url to the server
def train_url(url: str, score: float) -> int:
    print(f"Sending training data: {url} -> {score}")
    payload = json.dumps({"url": url, "score": score}).encode("utf-8")
    target = f"{BASE_URL}/correct_rank"
    try:
        data = make_request(target, data=payload, method="POST")
        print(format_response(data))
        return 0
    except urllib.error.URLError as err:
        print(f"Network error (is the server running on port 19091?): {err}")
        return 1
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
        return 1


# ##################################################################
# command rank
# handles the rank subcommand
def command_rank(args: argparse.Namespace) -> int:
    return rank_url(args.url)


# ##################################################################
# command train
# handles the train subcommand
def command_train(args: argparse.Namespace) -> int:
    return train_url(args.url, args.score)


# ##################################################################
# main
# parses arguments and dispatches to command handlers
def main(argv: list[str] = None) -> int:
    setproctitle.setproctitle("news-ranker-cli")

    parser = argparse.ArgumentParser(description="News Ranker CLI - Interact with the News Ranker Server")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    rank_parser = subparsers.add_parser("rank", help="Get the rank score for a URL")
    rank_parser.add_argument("url", help="URL of the article to score")
    rank_parser.set_defaults(func=command_rank)

    train_parser = subparsers.add_parser("train", help="Provide training feedback for a URL")
    train_parser.add_argument("url", help="URL of the article")
    train_parser.add_argument("score", type=float, help="Target score (1.0 - 10.0)")
    train_parser.set_defaults(func=command_train)

    args = parser.parse_args(argv)
    return args.func(args)


# ##################################################################
# entry point
# standard python pattern for running as script
if __name__ == "__main__":
    sys.exit(main())
