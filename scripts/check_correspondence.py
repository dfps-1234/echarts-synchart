import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='验证 PNG 与 JS 文件的一一对应关系')
    parser.add_argument('--png_dir', required=True, help='PNG 图片目录')
    parser.add_argument('--js_dir', required=True, help='增强后的 JS 文件目录')
    parser.add_argument('--check_log', action='store_true', help='是否同时检查 log 文件')
    args = parser.parse_args()

    png_dir = Path(args.png_dir)
    js_dir = Path(args.js_dir)

    png_files = list(png_dir.glob('*.png'))
    print(f"PNG 文件总数: {len(png_files)}")

    missing_js = []
    missing_log = []
    for png in png_files:
        base = png.stem
        js_file = js_dir / f"{base}.js"
        if not js_file.exists():
            missing_js.append(png.name)
        if args.check_log:
            log_file = png_dir / f"{base}.log"
            if not log_file.exists():
                missing_log.append(png.name)

    if missing_js:
        print(f"\n缺失对应 JS 文件的 PNG ({len(missing_js)} 个):")
        for f in missing_js[:20]:
            print(f"  {f}")
    else:
        print("\n所有 PNG 都有对应的 JS 文件。")

    if args.check_log and missing_log:
        print(f"\n缺失对应 LOG 文件的 PNG ({len(missing_log)} 个):")
        for f in missing_log[:20]:
            print(f"  {f}")
    elif args.check_log:
        print("\n所有 PNG 都有对应的 LOG 文件。")

if __name__ == '__main__':
    main()