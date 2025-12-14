"""File storage infrastructure."""

import hashlib
import shutil
from pathlib import Path
from typing import BinaryIO
from loguru import logger


class FileStorage:
    """File storage manager for uploaded documents."""

    def __init__(self, storage_path: str):
        """
        Initialize file storage.

        Args:
            storage_path: Base path for file storage
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"File storage initialized at: {self.storage_path}")

    def save_file(self, file: BinaryIO, collection_id: int, filename: str) -> tuple[str, str, int]:
        """
        Save uploaded file to storage.

        Args:
            file: File object to save
            collection_id: ID of the collection
            filename: Original filename

        Returns:
            Tuple of (file_path, content_hash, file_size)
        """
        # Create collection directory
        collection_dir = self.storage_path / f"collection_{collection_id}"
        collection_dir.mkdir(parents=True, exist_ok=True)

        # Generate file path
        file_path = collection_dir / filename

        # Handle duplicate filenames
        counter = 1
        while file_path.exists():
            stem = Path(filename).stem
            suffix = Path(filename).suffix
            file_path = collection_dir / f"{stem}_{counter}{suffix}"
            counter += 1

        # Save file and calculate hash
        hasher = hashlib.sha256()
        file_size = 0

        with open(file_path, "wb") as f:
            while chunk := file.read(8192):
                f.write(chunk)
                hasher.update(chunk)
                file_size += len(chunk)

        content_hash = hasher.hexdigest()
        relative_path = str(file_path.relative_to(self.storage_path))

        logger.info(f"✓ Saved file: {relative_path} ({file_size} bytes)")
        return relative_path, content_hash, file_size

    def get_file_path(self, relative_path: str) -> Path:
        """Get absolute file path from relative path."""
        return self.storage_path / relative_path

    def delete_file(self, relative_path: str) -> None:
        """Delete a file from storage."""
        file_path = self.get_file_path(relative_path)
        if file_path.exists():
            file_path.unlink()
            logger.info(f"✓ Deleted file: {relative_path}")

    def delete_collection_files(self, collection_id: int) -> None:
        """Delete all files for a collection."""
        collection_dir = self.storage_path / f"collection_{collection_id}"
        if collection_dir.exists():
            shutil.rmtree(collection_dir)
            logger.info(f"✓ Deleted collection directory: collection_{collection_id}")

