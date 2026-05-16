#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path
import shutil


PROMPT_PREFIX = (
    "Rewrite the following story summary into a detailed passage matching the user's preference, "
    "without any explanation before or after it: "
)


LLM_BOOKS = {
    "Moby_Dick": "adventure__Moby_Dick_output.txt",
    "Peter Pan": "adventure__Peter_Pan_output.txt",
    "Robinson Crusoe": "adventure__Robinson_Crusoe_output.txt",
    "The Adventures of Tom Sawyer": "adventure__The_Adventures_of_Tom_Sawyer_output.txt",
    "The Coral Island": "adventure__The_Coral_Island_output.txt",
}


RAG_BOOKS = [
    "A_Christmas_Carol",
    "A_Hero_of_Our_Timebook_3",
    "A_Hero_of_Our_Timebook_4",
    "A_Room_with_a_Viewpart_1",
    "A_Study_in_Scarletpart_i",
]


BOOK_METADATA = {
    "Moby_Dick": {
        "title": "Moby-Dick",
        "author": "Herman Melville",
        "genres": ["Classic", "Adventure", "Maritime Fiction"],
        "score": "0.00",
        "premise": "An obsessed captain drives the Pequod on a catastrophic voyage to hunt the white whale that maimed him.",
    },
    "Peter Pan": {
        "title": "Peter Pan",
        "author": "J. M. Barrie",
        "genres": ["Classic", "Fantasy", "Adventure"],
        "score": "0.00",
        "premise": "The Darling children are swept into Neverland, where eternal childhood, pirates, and peril collide.",
    },
    "Robinson Crusoe": {
        "title": "Robinson Crusoe",
        "author": "Daniel Defoe",
        "genres": ["Classic", "Adventure", "Survival"],
        "score": "0.00",
        "premise": "A restless young man defies his family, is shipwrecked, and struggles to survive in isolation for years.",
    },
    "The Adventures of Tom Sawyer": {
        "title": "The Adventures of Tom Sawyer",
        "author": "Mark Twain",
        "genres": ["Classic", "Adventure", "Coming-of-Age"],
        "score": "0.00",
        "premise": "A mischievous boy navigates friendship, freedom, danger, and moral awakening along the Mississippi.",
    },
    "The Coral Island": {
        "title": "The Coral Island",
        "author": "R. M. Ballantyne",
        "genres": ["Classic", "Adventure", "Survival"],
        "score": "0.00",
        "premise": "Three boys are stranded on a Pacific island and face shipwreck, conflict, and survival in an unfamiliar world.",
    },
    "A_Christmas_Carol": {
        "title": "A Christmas Carol",
        "author": "Charles Dickens",
        "genres": ["Classic", "Ghost Story", "Novella"],
        "score": "0.00",
        "premise": "A miser is confronted by spectral visitors who force him to reckon with his past, present, and future.",
    },
    "A_Hero_of_Our_Timebook_2": {
        "title": "A Hero of Our Time",
        "author": "Mikhail Lermontov",
        "genres": ["Classic", "Psychological Fiction", "Russian Literature"],
        "score": "0.00",
        "premise": "Episodes from Pechorin's life reveal a brilliant yet destructive antihero moving through love, conflict, and disillusionment.",
    },
    "A_Hero_of_Our_Timebook_3": {
        "title": "A Hero of Our Time",
        "author": "Mikhail Lermontov",
        "genres": ["Classic", "Psychological Fiction", "Russian Literature"],
        "score": "0.00",
        "premise": "Episodes from Pechorin's life reveal a brilliant yet destructive antihero moving through love, conflict, and disillusionment.",
    },
    "A_Hero_of_Our_Timebook_4": {
        "title": "A Hero of Our Time",
        "author": "Mikhail Lermontov",
        "genres": ["Classic", "Psychological Fiction", "Russian Literature"],
        "score": "0.00",
        "premise": "Episodes from Pechorin's life reveal a brilliant yet destructive antihero moving through love, conflict, and disillusionment.",
    },
    "A_Little_Princess": {
        "title": "A Little Princess",
        "author": "Frances Hodgson Burnett",
        "genres": ["Children", "Classic", "Coming-of-Age"],
        "score": "0.00",
        "premise": "A wealthy boarding-school girl loses her fortune and must preserve her dignity and kindness through hardship.",
    },
    "A_Room_with_a_Viewpart_1": {
        "title": "A Room with a View",
        "author": "E. M. Forster",
        "genres": ["Classic", "Romance", "Social Novel"],
        "score": "0.00",
        "premise": "A young Englishwoman abroad begins to question convention, propriety, and the life expected of her.",
    },
    "A_Study_in_Scarletpart_i": {
        "title": "A Study in Scarlet",
        "author": "Arthur Conan Doyle",
        "genres": ["Classic", "Detective Fiction", "Mystery"],
        "score": "0.00",
        "premise": "Dr. Watson meets Sherlock Holmes as they investigate a baffling murder marked by strange clues and hidden motives.",
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def prepare_llm(llm_output_dir: Path, target_root: Path) -> list[str]:
    prepared = []
    ensure_dir(target_root)
    for book_name, output_name in LLM_BOOKS.items():
        source_file = llm_output_dir / output_name
        if not source_file.exists():
            raise FileNotFoundError(f"missing llm output: {source_file}")
        book_dir = target_root / book_name
        ensure_dir(book_dir)
        shutil.copy2(source_file, book_dir / "llm_rewrite.txt")
        prepared.append(book_name)
    return prepared


def prepare_rag(rag_inputs_root: Path, target_root: Path) -> list[str]:
    prepared = []
    ensure_dir(target_root)
    for book_name in RAG_BOOKS:
        source_file = rag_inputs_root / book_name / "rewritten.txt"
        if not source_file.exists():
            raise FileNotFoundError(f"missing rag rewritten text: {source_file}")
        book_dir = target_root / book_name
        ensure_dir(book_dir)
        shutil.copy2(source_file, book_dir / "rag_rewrite.txt")
        prepared.append(book_name)
    return prepared


def prepare_pplug(pplug_jsonl: Path, target_root: Path) -> list[str]:
    prepared = []
    ensure_dir(target_root)

    grouped_predictions = defaultdict(list)
    with pplug_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            _, book_name, section_name = record["id"].split("::")
            grouped_predictions[book_name].append((section_name, record.get("prediction", "").strip()))

    for book_name, section_rows in sorted(grouped_predictions.items()):
        section_rows.sort(key=lambda row: row[0])
        content = []
        for section_name, prediction in section_rows:
            section_text = prediction if prediction else ""
            content.append(f"[{section_name}]\n{section_text}".strip())
        book_dir = target_root / book_name
        ensure_dir(book_dir)
        write_text(book_dir / "pplug_rewrite.txt", "\n\n".join(content))
        prepared.append(book_name)
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--llm_output_dir",
        default="/data1/YuhangFu/llm_baseline/experiment_results_first5_gemini_20260503_164400/gemini-3-pro-preview",
    )
    parser.add_argument(
        "--rag_inputs_root",
        default="/data1/YuhangFu/Evaluation/runs/codex_eval_mix_20260503_v1/_inputs/rag",
    )
    parser.add_argument(
        "--pplug_jsonl",
        default="/data1/YuhangFu/Representation_baseline/PPlug/PPlug/code/output_assets_73_chunk512_sentence_dec2_lr1e5/booksum_first5_gen/booksum_first5_generations.jsonl",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    rewritten_roots = output_root / "rewritten_roots"
    ensure_dir(rewritten_roots)

    llm_books = prepare_llm(Path(args.llm_output_dir), rewritten_roots / "llm")
    rag_books = prepare_rag(Path(args.rag_inputs_root), rewritten_roots / "rag")
    pplug_books = prepare_pplug(Path(args.pplug_jsonl), rewritten_roots / "pplug")

    metadata_path = output_root / "metadata_overrides.json"
    metadata_path.write_text(json.dumps(BOOK_METADATA, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "llm": llm_books,
        "rag": rag_books,
        "pplug": pplug_books,
        "metadata_overrides": str(metadata_path),
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"prepared llm books: {llm_books}")
    print(f"prepared rag books: {rag_books}")
    print(f"prepared pplug books: {pplug_books}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
