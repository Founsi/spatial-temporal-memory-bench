#!/usr/bin/env python3
"""
Human-in-the-Loop Review Server

A local web application for reviewing, editing, and approving
pipeline-generated annotation samples before they enter the benchmark.

Usage:
    python -m pipeline.review.server --samples data/curated/
    python -m pipeline.review.server --samples data/curated/kitchen_abc123.json --port 8765
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ReviewServer:
    """Manages sample data and serves the review UI."""

    def __init__(self, samples_path: Path, project_root: Path):
        self.project_root = project_root
        self.samples: dict[str, dict] = {}
        self.samples_dir: Optional[Path] = None
        self._load_samples(samples_path)

    def _load_samples(self, path: Path):
        """Load one or more annotation JSON files."""
        if path.is_file():
            self._load_single(path)
            self.samples_dir = path.parent
        elif path.is_dir():
            self.samples_dir = path
            for f in sorted(path.glob("*.json")):
                self._load_single(f)
        else:
            raise FileNotFoundError(f"Samples path not found: {path}")

        if not self.samples:
            raise FileNotFoundError(f"No annotation JSON files found in: {path}")

        print(f"Loaded {len(self.samples)} sample(s) for review")

    def _load_single(self, path: Path):
        """Load a single annotation file."""
        with open(path) as f:
            data = json.load(f)
        sample_id = data.get("id", path.stem)
        # Inject review field if missing
        if "review" not in data:
            data["review"] = {
                "status": "pending",
                "reviewer": "",
                "review_date": "",
                "frame_reviews": {},
                "qa_reviews": {},
                "notes": "",
            }
        data["_source_path"] = str(path)
        self.samples[sample_id] = data

    def save_sample(self, sample_id: str):
        """Save a sample back to disk."""
        sample = self.samples[sample_id]
        path = Path(sample["_source_path"])
        # Remove internal field before saving
        save_data = {k: v for k, v in sample.items() if not k.startswith("_")}
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

    def get_sample_list(self) -> list[dict]:
        """Return summary of all samples."""
        result = []
        for sid, sample in self.samples.items():
            review = sample.get("review", {})
            result.append({
                "id": sid,
                "environment": sample.get("visual_sequence", {}).get("environment", "?"),
                "num_frames": len(sample.get("visual_sequence", {}).get("frames", [])),
                "num_qa": len(sample.get("qa_pairs", [])),
                "status": review.get("status", "pending"),
                "reviewer": review.get("reviewer", ""),
            })
        return result


def create_handler(review_server: ReviewServer):
    """Create a request handler class bound to our ReviewServer."""

    class ReviewHandler(SimpleHTTPRequestHandler):
        server_instance = review_server

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path

            if path == "/" or path == "/index.html":
                self._serve_ui()
            elif path == "/api/samples":
                self._json_response(self.server_instance.get_sample_list())
            elif path.startswith("/api/sample/"):
                sample_id = path.split("/api/sample/")[1]
                if sample_id in self.server_instance.samples:
                    sample = self.server_instance.samples[sample_id]
                    # Strip internal fields
                    clean = {k: v for k, v in sample.items() if not k.startswith("_")}
                    self._json_response(clean)
                else:
                    self._json_response({"error": "Sample not found"}, 404)
            elif path.startswith("/images/"):
                self._serve_image(path)
            else:
                self.send_error(404)

        def do_POST(self):
            parsed = urlparse(self.path)
            path = parsed.path

            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}

            if path.startswith("/api/sample/") and path.endswith("/save"):
                sample_id = path.replace("/api/sample/", "").replace("/save", "")
                self._handle_save(sample_id, data)
            elif path.startswith("/api/sample/") and path.endswith("/review"):
                sample_id = path.replace("/api/sample/", "").replace("/review", "")
                self._handle_review(sample_id, data)
            else:
                self.send_error(404)

        def _handle_save(self, sample_id: str, data: dict):
            """Save edited sample data."""
            if sample_id not in self.server_instance.samples:
                self._json_response({"error": "Sample not found"}, 404)
                return
            sample = self.server_instance.samples[sample_id]
            # Update allowed fields
            if "visual_sequence" in data:
                sample["visual_sequence"] = data["visual_sequence"]
            if "qa_pairs" in data:
                sample["qa_pairs"] = data["qa_pairs"]
            if "review" in data:
                sample["review"] = data["review"]
            self.server_instance.save_sample(sample_id)
            self._json_response({"status": "saved"})

        def _handle_review(self, sample_id: str, data: dict):
            """Update review status."""
            if sample_id not in self.server_instance.samples:
                self._json_response({"error": "Sample not found"}, 404)
                return
            sample = self.server_instance.samples[sample_id]
            review = sample.setdefault("review", {})
            review.update(data)
            if "review_date" not in data:
                review["review_date"] = date.today().isoformat()
            self.server_instance.save_sample(sample_id)
            self._json_response({"status": "updated", "review": review})

        def _serve_ui(self):
            """Serve the review UI HTML."""
            ui_path = Path(__file__).parent / "ui.html"
            if not ui_path.exists():
                self.send_error(500, "UI template not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(ui_path.read_bytes())

        def _serve_image(self, path: str):
            """Serve an image file from the project."""
            # Strip leading /images/ and resolve relative to project root / data
            rel_path = path.lstrip("/images/")
            candidates = [
                self.server_instance.project_root / rel_path,
                self.server_instance.project_root / "data" / rel_path,
            ]
            if self.server_instance.samples_dir:
                candidates.append(self.server_instance.samples_dir / rel_path)
                candidates.append(self.server_instance.samples_dir.parent / rel_path)

            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    self.send_response(200)
                    suffix = candidate.suffix.lower()
                    ct = {
                        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".webp": "image/webp",
                    }.get(suffix, "application/octet-stream")
                    self.send_header("Content-Type", ct)
                    self.end_headers()
                    self.wfile.write(candidate.read_bytes())
                    return

            self.send_error(404, f"Image not found: {rel_path}")

        def _json_response(self, data, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode("utf-8"))

        def log_message(self, format, *args):
            """Suppress default logging for cleaner output."""
            if "/api/" in str(args[0]) if args else False:
                return
            super().log_message(format, *args)

    return ReviewHandler


def main():
    parser = argparse.ArgumentParser(description="Launch the annotation review UI")
    parser.add_argument(
        "--samples", required=True,
        help="Path to annotation JSON file or directory of JSON files"
    )
    parser.add_argument("--port", type=int, default=8765, help="Server port (default: 8765)")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    review_server = ReviewServer(Path(args.samples), project_root)
    handler_class = create_handler(review_server)

    httpd = HTTPServer((args.host, args.port), handler_class)
    print(f"\n  Review UI running at: http://{args.host}:{args.port}")
    print(f"  Reviewing {len(review_server.samples)} sample(s)")
    print(f"  Press Ctrl+C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down review server.")
        httpd.shutdown()


if __name__ == "__main__":
    main()
