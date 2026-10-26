"""
The database: patients, nights and their recordings, model files, analyses and a log of actions.

Each table is a SQLModel class: SQLModel creates the tables and turns the Python expressions of
store.py into SQL. Every patient's data is here, the recordings included: a night's details are a
row of `recordings`, and the content of its .csv file a row of `recording_files`.

No web code here, and no `from __future__ import annotations`: SQLModel reads the annotations
below at run time to build the relationships between tables.
"""
import os
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, LargeBinary
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine

DB_URL = os.environ.get("DB_URL", "sqlite:///apnea.sqlite")
# In Docker, PostgreSQL: postgresql://<user>:<password>@database:5432/<database>, through psycopg,
# SQLAlchemy's default PostgreSQL driver.
# Without the variable, a SQLite file in the current folder, for running the service on its own.


def now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(**kwargs):
    return Field(sa_type=DateTime(timezone=True), **kwargs)


class Patient(SQLModel, table=True):
    __tablename__ = "patients"

    id: int | None = Field(default=None, primary_key=True)
    code: str = Field(max_length=64)                           # the name; several patients may share one
    created_at: datetime = _timestamp(default_factory=now)

    recordings: list["Recording"] = Relationship(back_populates="patient")


class Recording(SQLModel, table=True):
    """One night: the details of a .csv file uploaded or imported for a patient."""
    __tablename__ = "recordings"

    id: int | None = Field(default=None, primary_key=True)
    patient_id: int = Field(foreign_key="patients.id", index=True)
    file_name: str                                              # the original file's name
    sha256: str = Field(index=True, max_length=64)               # checksum of the content
    size_bytes: int = Field(sa_type=BigInteger)
    mtime_ns: int = Field(sa_type=BigInteger)                   # the original file's modification time
    samples: int
    sample_rate_hz: float
    recorded_at: datetime = _timestamp()                        # the same time: sets the night order
    registered_at: datetime = _timestamp(default_factory=now)

    patient: Patient = Relationship(back_populates="recordings")
    analyses: list["Analysis"] = Relationship(back_populates="recording")


class RecordingFile(SQLModel, table=True):
    """The content of a night's .csv file, byte for byte, apart from its details."""
    __tablename__ = "recording_files"

    recording_id: int = Field(foreign_key="recordings.id", primary_key=True)
    content: bytes = Field(sa_type=LargeBinary)
    # A table of its own, so listing nights never reads the contents (about 20 MB per night).


class ModelFile(SQLModel, table=True):
    """One version of a .tflite file, told apart by its checksum."""
    __tablename__ = "models"

    id: int | None = Field(default=None, primary_key=True)
    file_name: str
    sha256: str = Field(unique=True, max_length=64)
    size_bytes: int = Field(sa_type=BigInteger)
    quantized: bool
    first_seen: datetime = _timestamp(default_factory=now)


class Analysis(SQLModel, table=True):
    """One run of a model on a night. Never changed once written: a new run adds a new row."""
    __tablename__ = "analyses"

    id: int | None = Field(default=None, primary_key=True)
    recording_id: int = Field(foreign_key="recordings.id", index=True)
    model_file_id: int = Field(foreign_key="models.id")
    threshold: float
    smooth_window: int
    minutes: int
    apnea_minutes: int
    minutes_per_hour: float
    service_version: str
    created_at: datetime = _timestamp(default_factory=now)

    recording: Recording = Relationship(back_populates="analyses")
    model_file: ModelFile = Relationship()
    per_minute: list["AnalysisMinute"] = Relationship(
        sa_relationship_kwargs={"order_by": "AnalysisMinute.minute"})


class AnalysisMinute(SQLModel, table=True):
    """One minute of an analysis: what the chart draws."""
    __tablename__ = "analysis_minutes"

    analysis_id: int = Field(foreign_key="analyses.id", primary_key=True)
    minute: int = Field(primary_key=True)                       # 0 = the first minute of the night
    prob: float
    prob_smoothed: float
    apnea: bool


class Event(SQLModel, table=True):
    """What happened and when: patient created, night uploaded or imported, night analysed."""
    __tablename__ = "events"

    id: int | None = Field(default=None, primary_key=True)
    at: datetime = _timestamp(default_factory=now)
    action: str
    patient_id: int | None = Field(default=None, foreign_key="patients.id")
    recording_id: int | None = Field(default=None, foreign_key="recordings.id")
    analysis_id: int | None = Field(default=None, foreign_key="analyses.id")
    detail: str | None = None


engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
# pool_pre_ping tests a pooled connection before using it, so a restarted database does not
# fail the next request. check_same_thread: SQLite's own guard, off because FastAPI runs requests
# in a pool of threads.


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    # Creates the tables that do not exist yet; existing tables and their rows are left as they are.


def get_session():
    with Session(engine) as session:
        yield session
    # A FastAPI dependency: one session per request, closed when the request ends.
