"""Service for sensor management."""

import csv
import io
from typing import BinaryIO

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.domain.schemas import SensorListResponse, SensorResponse
from src.api.infrastructure.repositories import CollectionRepository, SensorRepository


class SensorService:
    """Service for managing sensors in collections."""

    async def bulk_upsert(
        self,
        session: AsyncSession,
        collection_id: int,
        sensors_data: list[dict],
    ) -> SensorListResponse:
        """
        Bulk upsert sensors for a collection.
        
        Args:
            session: Database session
            collection_id: Collection ID
            sensors_data: List of sensor dicts with: sensor_id, name, description, unit, example
        
        Returns:
            List of upserted sensors with total count
        """
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Upsert sensors
        sensor_repo = SensorRepository(session)
        sensors = await sensor_repo.bulk_upsert(collection_id, sensors_data)
        
        await session.commit()

        logger.info(f"✓ Upserted {len(sensors)} sensors for collection {collection_id}")

        return SensorListResponse(
            sensors=[self._sensor_to_response(s) for s in sensors],
            total=len(sensors)
        )

    async def import_from_csv(
        self,
        session: AsyncSession,
        collection_id: int,
        file: BinaryIO,
    ) -> SensorListResponse:
        """
        Import sensors from CSV file.
        
        Expected CSV format:
        id,name,description,unit,example
        14A3003I,Propane/Propylene at Bottom,Inferred volume percentage,...,%,0.67
        
        Args:
            session: Database session
            collection_id: Collection ID
            file: CSV file (binary mode)
        
        Returns:
            List of imported sensors with total count
        """
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        # Parse CSV
        try:
            content = file.read().decode('utf-8')
            csv_reader = csv.DictReader(io.StringIO(content))
            
            sensors_data = []
            for row in csv_reader:
                # Map CSV columns to sensor fields
                # CSV has: id, name, description, unit, example
                # We need: sensor_id, name, description, unit, example
                sensor_data = {
                    "sensor_id": row.get("id", "").strip(),
                    "name": row.get("name", "").strip(),
                    "description": row.get("description", "").strip() or None,
                    "unit": row.get("unit", "").strip() or None,
                    "example": row.get("example", "").strip() or None,
                }
                
                # Validate required fields
                if not sensor_data["sensor_id"] or not sensor_data["name"]:
                    logger.warning(f"Skipping invalid row: {row}")
                    continue
                
                sensors_data.append(sensor_data)
            
            if not sensors_data:
                raise ValueError("No valid sensors found in CSV file")
            
            logger.info(f"📄 Parsed {len(sensors_data)} sensors from CSV")
            
        except Exception as e:
            logger.error(f"Failed to parse CSV: {e}")
            raise ValueError(f"Invalid CSV format: {str(e)}")

        # Upsert sensors
        sensor_repo = SensorRepository(session)
        sensors = await sensor_repo.bulk_upsert(collection_id, sensors_data)
        
        await session.commit()

        logger.info(f"✓ Imported {len(sensors)} sensors for collection {collection_id}")

        return SensorListResponse(
            sensors=[self._sensor_to_response(s) for s in sensors],
            total=len(sensors)
        )

    async def list_sensors(
        self,
        session: AsyncSession,
        collection_id: int,
    ) -> SensorListResponse:
        """List all sensors for a collection."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        sensor_repo = SensorRepository(session)
        sensors = await sensor_repo.list_by_collection(collection_id)

        return SensorListResponse(
            sensors=[self._sensor_to_response(s) for s in sensors],
            total=len(sensors)
        )

    async def get_sensor(
        self,
        session: AsyncSession,
        collection_id: int,
        sensor_db_id: int,
    ) -> SensorResponse:
        """Get a specific sensor by database ID."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        sensor_repo = SensorRepository(session)
        sensor = await sensor_repo.get_by_id(sensor_db_id)

        if not sensor or sensor.collection_id != collection_id:
            raise ValueError(f"Sensor with ID {sensor_db_id} not found in collection {collection_id}")

        return self._sensor_to_response(sensor)

    async def delete_sensor(
        self,
        session: AsyncSession,
        collection_id: int,
        sensor_db_id: int,
    ) -> None:
        """Delete a specific sensor."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        sensor_repo = SensorRepository(session)
        sensor = await sensor_repo.get_by_id(sensor_db_id)

        if not sensor or sensor.collection_id != collection_id:
            raise ValueError(f"Sensor with ID {sensor_db_id} not found in collection {collection_id}")

        await sensor_repo.delete(sensor)
        await session.commit()

        logger.info(f"✓ Deleted sensor {sensor_db_id} from collection {collection_id}")

    async def delete_all_sensors(
        self,
        session: AsyncSession,
        collection_id: int,
    ) -> dict:
        """Delete all sensors for a collection."""
        # Verify collection exists
        collection_repo = CollectionRepository(session)
        collection = await collection_repo.get_by_id(collection_id)
        if not collection:
            raise ValueError(f"Collection with ID {collection_id} not found")

        sensor_repo = SensorRepository(session)
        count = await sensor_repo.delete_all_by_collection(collection_id)
        await session.commit()

        logger.info(f"✓ Deleted {count} sensors from collection {collection_id}")

        return {"deleted_count": count, "message": f"Deleted {count} sensors"}

    def _sensor_to_response(self, sensor) -> SensorResponse:
        """Convert sensor entity to response schema."""
        return SensorResponse(
            id=sensor.id,
            collection_id=sensor.collection_id,
            sensor_id=sensor.sensor_id,
            name=sensor.name,
            description=sensor.description,
            unit=sensor.unit,
            example=sensor.example,
            created_at=sensor.created_at,
            updated_at=sensor.updated_at,
        )

