"""Read-only queries against the existing local CompanyX audit restoration."""

from io import StringIO
import subprocess

import pandas as pd


def query(statement: str) -> pd.DataFrame:
    """Use the container's existing credential without returning it to the host."""
    command = [
        "docker", "exec", "-i", "companyx-audit", "sh", "-c",
        'export SQLCMDPASSWORD="$MSSQL_SA_PASSWORD"; '
        'exec /opt/mssql-tools18/bin/sqlcmd '
        '-S tcp:127.0.0.1,1433 -U sa -C -I -b -l 10 '
        '-d CompanyX_Audit -s "|" -W -w 65535',
    ]
    result = subprocess.run(
        command, input="SET NOCOUNT ON;\n" + statement,
        text=True, capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError("CompanyX read-only query failed: " + result.stdout + result.stderr)
    return pd.read_csv(
        StringIO(result.stdout.strip()), sep="|", skiprows=[1], na_values=["NULL"]
    )
