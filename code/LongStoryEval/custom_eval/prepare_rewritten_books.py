#!/usr/bin/env python3
import argparse
import fnmatch
import json
import math
import os
import re
from pathlib import Path


def count_words(text: str) -> int:
    return len(text.split())


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    return text.strip()


def split_paragraphs(text: str):
    paragraphs = []
    for part in re.split(r"\n\s*\n", normalize_text(text)):
        cleaned = "\n".join(line.strip() for line in part.split("\n") if line.strip())
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def humanize_book_name(book_name: str) -> str:
    return book_name.replace("_", " ").strip()


def load_json_if_exists(path: str):
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"metadata overrides not found: {path}")
    return json.load(open(p, "r", encoding="utf-8"))


def collect_original_chapters(original_book_dir: Path):
    chapter_files = sorted(
        [p for p in original_book_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"],
        key=lambda p: p.name,
    )
    if not chapter_files:
        raise ValueError(f"no chapter txt files found in {original_book_dir}")
    chapters = []
    for chapter_file in chapter_files:
        content = normalize_text(chapter_file.read_text(encoding="utf-8", errors="ignore"))
        chapters.append(
            {
                "name": chapter_file.stem,
                "text": content,
                "word_count": max(count_words(content), 1),
            }
        )
    return chapters


def allocate_paragraphs(paragraphs, target_word_counts):
    if not paragraphs:
        return [[] for _ in target_word_counts]

    para_word_counts = [max(count_words(p), 1) for p in paragraphs]
    total_target = sum(target_word_counts)
    total_rewritten = sum(para_word_counts)
    scaled_targets = []
    for target in target_word_counts:
        scaled = max(1, int(round(total_rewritten * (target / max(total_target, 1)))))
        scaled_targets.append(scaled)

    assignments = []
    para_index = 0
    paragraph_total = len(paragraphs)
    chapter_total = len(scaled_targets)

    for chapter_index, target_words in enumerate(scaled_targets):
        remaining_chapters = chapter_total - chapter_index
        remaining_paragraphs = paragraph_total - para_index
        min_keep = max(remaining_chapters - 1, 0)

        if chapter_index == chapter_total - 1:
            assignments.append(paragraphs[para_index:])
            break

        current = []
        current_words = 0
        while para_index < paragraph_total:
            if (paragraph_total - (para_index + 1)) < min_keep:
                break

            current.append(paragraphs[para_index])
            current_words += para_word_counts[para_index]
            para_index += 1

            if current_words >= target_words and current:
                break

        if not current and para_index < paragraph_total:
            current = [paragraphs[para_index]]
            para_index += 1

        assignments.append(current)

    if len(assignments) < chapter_total:
        assignments.extend([[] for _ in range(chapter_total - len(assignments))])
    return assignments


def build_output_id(book_name: str, rewritten_file: Path) -> str:
    stem = rewritten_file.stem
    suffix = stem
    match = re.match(r"modeled_user_data_(.+)", stem)
    if match:
        suffix = match.group(1)
    return f"{book_name}__{suffix}"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rewritten_root", type=str, required=True)
    parser.add_argument("--original_root", type=str, required=True)
    parser.add_argument("--output_books_dir", type=str, required=True)
    parser.add_argument("--output_book_info", type=str, required=True)
    parser.add_argument("--output_book_ids", type=str, required=True)
    parser.add_argument("--metadata_overrides", type=str, default="")
    parser.add_argument("--books", nargs="*", default=[])
    parser.add_argument("--file_patterns", nargs="*", default=["*.txt"])
    args = parser.parse_args()

    rewritten_root = Path(args.rewritten_root)
    original_root = Path(args.original_root)
    output_books_dir = Path(args.output_books_dir)
    output_book_info = Path(args.output_book_info)
    output_book_ids = Path(args.output_book_ids)
    ensure_dir(output_books_dir)
    ensure_dir(output_book_info.parent)
    ensure_dir(output_book_ids.parent)

    metadata_overrides = load_json_if_exists(args.metadata_overrides)
    selected_books = set(args.books) if args.books else None

    all_book_info = {}
    created_ids = []

    for book_dir in sorted([p for p in rewritten_root.iterdir() if p.is_dir()], key=lambda p: p.name):
        if selected_books and book_dir.name not in selected_books:
            continue

        original_book_dir = original_root / book_dir.name
        if not original_book_dir.exists():
            print(f"skip {book_dir.name}: original directory not found at {original_book_dir}")
            continue

        original_chapters = collect_original_chapters(original_book_dir)
        target_word_counts = [chapter["word_count"] for chapter in original_chapters]
        book_overrides = metadata_overrides.get(book_dir.name, {})

        candidate_files = []
        for child in sorted(book_dir.iterdir(), key=lambda p: p.name):
            if not child.is_file():
                continue
            if any(fnmatch.fnmatch(child.name, pattern) for pattern in args.file_patterns):
                candidate_files.append(child)

        for rewritten_file in candidate_files:
            rewritten_text = normalize_text(rewritten_file.read_text(encoding="utf-8", errors="ignore"))
            paragraphs = split_paragraphs(rewritten_text)
            if not paragraphs:
                print(f"skip {rewritten_file}: empty rewritten text")
                continue

            allocations = allocate_paragraphs(paragraphs, target_word_counts)
            output_id = build_output_id(book_dir.name, rewritten_file)
            output_json = {
                "book_info": [],
                "chaps_info": [],
                "chaps": [],
            }

            for chapter_info, assigned_paragraphs in zip(original_chapters, allocations):
                if not assigned_paragraphs:
                    assigned_paragraphs = [""]
                output_json["chaps"].append(
                    {
                        "title": chapter_info["name"],
                        "content": assigned_paragraphs,
                    }
                )

            (output_books_dir / f"{output_id}.json").write_text(
                json.dumps(output_json, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            all_book_info[output_id] = {
                "title": book_overrides.get("title", humanize_book_name(book_dir.name)),
                "author": book_overrides.get("author", "unknown"),
                "genres": book_overrides.get("genres", ["Unknown"]),
                "score": book_overrides.get("score", "0.00"),
                "premise": book_overrides.get("premise", ""),
                "source_book": book_dir.name,
                "source_file": rewritten_file.name,
                "original_chapter_count": len(original_chapters),
                "rewritten_paragraph_count": len(paragraphs),
            }
            created_ids.append(output_id)
            print(f"prepared {output_id}")

    output_book_info.write_text(json.dumps(all_book_info, ensure_ascii=False, indent=2), encoding="utf-8")
    output_book_ids.write_text("\n".join(created_ids) + ("\n" if created_ids else ""), encoding="utf-8")
    print(f"wrote {len(created_ids)} books to {output_books_dir}")


if __name__ == "__main__":
    main()
