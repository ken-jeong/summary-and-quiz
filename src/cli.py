"""Command-line interface for Summary and Quiz Generator."""

import argparse
import sys
import logging
from pathlib import Path

from .generator import SummaryQuizGenerator
from .config import Config


def setup_logging(verbose: bool) -> None:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def read_input(input_path: str | None) -> str:
    """Read input text from file or stdin.

    Args:
        input_path: Path to input file, or None for stdin

    Returns:
        Input text content
    """
    if input_path:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        return path.read_text(encoding="utf-8")

    print("텍스트를 입력하세요 (Ctrl+D로 종료):")
    return sys.stdin.read()


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="한국어 텍스트 요약 및 퀴즈 생성 AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 텍스트 요약
  python -m src.cli --mode summary --input article.txt

  # 퀴즈 생성
  python -m src.cli --mode quiz --input summary.txt

  # 요약 + 퀴즈 동시 생성
  python -m src.cli --mode both --input article.txt

  # 표준 입력에서 읽기
  echo "텍스트..." | python -m src.cli --mode summary
        """
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["summary", "quiz", "both"],
        default="both",
        help="생성 모드: summary(요약), quiz(퀴즈), both(둘 다)"
    )

    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="입력 파일 경로 (없으면 표준 입력 사용)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="출력 파일 경로 (없으면 표준 출력)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="사용할 모델 이름 (기본: Bllossom/llama-3.2-Korean-Bllossom-AICA-5B)"
    )

    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.1,
        help="생성 temperature (기본: 0.1, 높을수록 다양한 출력)"
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="최대 생성 토큰 수 (기본: 256)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로깅 출력"
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Read input text
        text = read_input(args.input)
        if not text.strip():
            logger.error("입력 텍스트가 비어있습니다.")
            return 1

        # Configure generator
        config = Config(
            temperature=args.temperature,
            max_new_tokens=args.max_tokens
        )

        if args.model:
            config.model_name = args.model

        # Generate output
        with SummaryQuizGenerator(config) as generator:
            if args.mode == "summary":
                result = f"[요약]\n{generator.summarize(text)}"

            elif args.mode == "quiz":
                result = f"[퀴즈]\n{generator.create_quiz(text)}"

            else:  # both
                output = generator.summarize_and_quiz(text)
                result = f"[요약]\n{output['summary']}\n\n[퀴즈]\n{output['quiz']}"

        # Write output
        if args.output:
            Path(args.output).write_text(result, encoding="utf-8")
            logger.info(f"결과가 {args.output}에 저장되었습니다.")
        else:
            print("\n" + "=" * 50)
            print(result)
            print("=" * 50)

        return 0

    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        return 1

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
