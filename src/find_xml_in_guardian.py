"""Scan all inner zips in GuardianObserver.zip looking for XML files.

Reads the outer zip's central directory from S3 via byte-range requests,
then iterates each inner zip in order, checking its file listing for any
.xml files. Prints nothing until an XML is found. Exits on first XML found.
"""

import boto3
import io
import struct
import sys
import zipfile

BUCKET     = "idiom-index"
KEY        = "raw/guardian/GuardianObserver.zip"
TOTAL_SIZE = 231256377493

s3 = boto3.client("s3")


def s3_get(start: int, end: int) -> bytes:
    resp = s3.get_object(Bucket=BUCKET, Key=KEY, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def read_central_directory() -> list[tuple[str, int, int]]:
    """Return list of (filename, local_offset, compressed_size) for all entries."""
    # ZIP64 EOCD locator
    tail = s3_get(TOTAL_SIZE - 131072, TOTAL_SIZE - 1)
    loc_idx = tail.rfind(b"PK\x06\x07")
    z64_offset = struct.unpack_from("<Q", tail, loc_idx + 8)[0]

    z64 = s3_get(z64_offset, z64_offset + 4095)
    cd_size   = struct.unpack_from("<Q", z64, 40)[0]
    cd_offset = struct.unpack_from("<Q", z64, 48)[0]

    cd_data = s3_get(cd_offset, cd_offset + cd_size - 1)

    entries = []
    pos = 0
    while pos < len(cd_data) - 4:
        if cd_data[pos:pos+4] != b"PK\x01\x02":
            break
        comp_size   = struct.unpack_from("<I", cd_data, pos+20)[0]
        fname_len   = struct.unpack_from("<H", cd_data, pos+28)[0]
        extra_len   = struct.unpack_from("<H", cd_data, pos+30)[0]
        comment_len = struct.unpack_from("<H", cd_data, pos+32)[0]
        loc_offset  = struct.unpack_from("<I", cd_data, pos+42)[0]
        fname = cd_data[pos+46:pos+46+fname_len].decode("utf-8", errors="replace")

        # ZIP64 extended info
        extra = cd_data[pos+46+fname_len:pos+46+fname_len+extra_len]
        ep = 0
        while ep < len(extra) - 4:
            tag = struct.unpack_from("<H", extra, ep)[0]
            sz  = struct.unpack_from("<H", extra, ep+2)[0]
            if tag == 0x0001:
                vals = [struct.unpack_from("<Q", extra, ep+4+i*8)[0] for i in range(sz//8)]
                if comp_size  == 0xFFFFFFFF and vals: comp_size  = vals.pop(0)
                if loc_offset == 0xFFFFFFFF and vals: loc_offset = vals.pop(0)
            ep += 4 + sz

        entries.append((fname, loc_offset, comp_size))
        pos += 46 + fname_len + extra_len + comment_len

    return entries


def fetch_inner_zip(loc_offset: int, comp_size: int) -> zipfile.ZipFile:
    """Download one inner zip from S3 by byte range and return as ZipFile."""
    # Read local file header to find data start
    lfh = s3_get(loc_offset, loc_offset + 299)
    fname_len = struct.unpack_from("<H", lfh, 26)[0]
    extra_len = struct.unpack_from("<H", lfh, 28)[0]
    data_start = loc_offset + 30 + fname_len + extra_len

    data = s3_get(data_start, data_start + comp_size - 1)
    return zipfile.ZipFile(io.BytesIO(data))


def main() -> None:
    entries = read_central_directory()
    inner_zips = [(f, off, sz) for f, off, sz in entries if f.endswith(".zip")]

    for i, (fname, offset, comp_size) in enumerate(reversed(inner_zips), 1):
        try:
            inner = fetch_inner_zip(offset, comp_size)
        except Exception as e:
            print(f"ERROR opening {fname}: {e}", file=sys.stderr)
            continue

        xml_files = [n for n in inner.namelist() if n.lower().endswith(".xml")]
        if xml_files:
            print(f"\nFound XML in inner zip #{i}: {fname}")
            for x in xml_files:
                print(f"  {x}")

            # Print a sample of the first XML
            sample = inner.read(xml_files[0])
            print(f"\n--- Sample of {xml_files[0]} ({len(sample)} bytes) ---")
            print(sample[:2000].decode("utf-8", errors="replace"))
            sys.exit(0)


if __name__ == "__main__":
    main()
