import argparse
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dist", nargs="+", type=Path)
    parser.add_argument("--tag")
    return parser.parse_args()


def archive_names(path):
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return archive.namelist()
    with tarfile.open(path) as archive:
        return archive.getnames()


def wheel_version(path):
    with zipfile.ZipFile(path) as archive:
        metadata = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        message = BytesParser().parsebytes(archive.read(metadata))
    return message["Version"]


def main():
    args = parse_args()
    wheels = [path for path in args.dist if path.suffix == ".whl"]
    sdists = [path for path in args.dist if path.name.endswith(".tar.gz")]
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one sdist")

    for path in args.dist:
        for name in archive_names(path):
            parts = PurePosixPath(name).parts
            if "examples" in parts or "test" in parts:
                raise SystemExit(f"Unexpected path in {path.name}: {name}")

    version = wheel_version(wheels[0])
    if f"-{version}.tar.gz" not in sdists[0].name:
        raise SystemExit("Wheel and sdist versions do not match")
    if args.tag is not None and args.tag.removeprefix("v") != version:
        raise SystemExit(f"Tag {args.tag!r} does not match package version {version!r}")

    print(f"Validated GeoTorch {version} wheel and sdist")


if __name__ == "__main__":
    main()
