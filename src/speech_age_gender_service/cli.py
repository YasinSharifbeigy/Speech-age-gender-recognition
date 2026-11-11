import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Speech age and gender recognition")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("download", help="Download checkpoint and preprocessing configuration")
    predict = commands.add_parser("predict", help="Analyze a single-speaker recording")
    predict.add_argument("audio", type=Path)
    predict.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    serve = commands.add_parser("serve", help="Start the local HTTP service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.command == "download":
        from .engine import download_models
        print(json.dumps(download_models(), indent=2))
    elif args.command == "predict":
        from .engine import Engine
        print(json.dumps(Engine(args.device).predict(args.audio), indent=2))
    else:
        import uvicorn
        uvicorn.run("speech_age_gender_service.api:app", host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
