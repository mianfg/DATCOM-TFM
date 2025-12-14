"""API routes for sensor management."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.application.sensor_service import SensorService
from src.api.domain.schemas import SensorBulkUpsert, SensorListResponse, SensorResponse
from src.api.infrastructure.database import Database
from src.api.infrastructure.container import get_container

router = APIRouter(prefix="/api/collections/{collection_id}/sensors", tags=["sensors"])


def get_db_session():
    """Get database session dependency."""
    container = get_container()
    db = container.database()
    return db.get_async_session()


def get_sensor_service():
    """Get sensor service dependency."""
    return SensorService()


@router.post("", response_model=SensorListResponse, status_code=201)
async def bulk_upsert_sensors(
    collection_id: int,
    data: SensorBulkUpsert,
    session: AsyncSession = Depends(get_db_session),
    service: SensorService = Depends(get_sensor_service),
):
    """
    Bulk upsert sensors for a collection.
    
    Upserts sensors by sensor_id:
    - If sensor_id exists in collection, updates it
    - If sensor_id is new, creates it
    
    This allows you to add/update multiple sensors at once.
    """
    try:
        async with session:
            sensors_data = [sensor.model_dump() for sensor in data.sensors]
            return await service.bulk_upsert(session, collection_id, sensors_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upsert sensors: {str(e)}")


@router.post("/import", response_model=SensorListResponse, status_code=201)
async def import_sensors_from_csv(
    collection_id: int,
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
    service: SensorService = Depends(get_sensor_service),
):
    """
    Import sensors from CSV file.
    
    Expected CSV format:
    ```csv
    id,name,description,unit,example
    14A3003I,Propane/Propylene at Bottom,Inferred volume percentage,...,%,0.67
    14AI1003,Propane at Bottom,Mole percentage of propane at bottom tray,%,0.53
    ```
    
    - **id**: Sensor identifier (required, must be unique per collection)
    - **name**: Sensor name (required)
    - **description**: Detailed description (optional)
    - **unit**: Measurement unit (optional)
    - **example**: Example value (optional)
    
    The import does upsert: updates existing sensors, creates new ones.
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        async with session:
            return await service.import_from_csv(session, collection_id, file.file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import CSV: {str(e)}")


@router.get("", response_model=SensorListResponse)
async def list_sensors(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: SensorService = Depends(get_sensor_service),
):
    """List all sensors for a collection."""
    try:
        async with session:
            return await service.list_sensors(session, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{sensor_id}", response_model=SensorResponse)
async def get_sensor(
    collection_id: int,
    sensor_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: SensorService = Depends(get_sensor_service),
):
    """Get a specific sensor by ID."""
    try:
        async with session:
            return await service.get_sensor(session, collection_id, sensor_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{sensor_id}", status_code=204)
async def delete_sensor(
    collection_id: int,
    sensor_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: SensorService = Depends(get_sensor_service),
):
    """Delete a specific sensor."""
    try:
        async with session:
            await service.delete_sensor(session, collection_id, sensor_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("", status_code=200)
async def delete_all_sensors(
    collection_id: int,
    session: AsyncSession = Depends(get_db_session),
    service: SensorService = Depends(get_sensor_service),
):
    """
    Delete all sensors for a collection.
    
    ⚠️ Warning: This will permanently delete all sensors in the collection.
    """
    try:
        async with session:
            return await service.delete_all_sensors(session, collection_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

