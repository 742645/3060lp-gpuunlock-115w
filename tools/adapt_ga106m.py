"""Add a default-off GA106M 2520 experiment to the pinned upstream checkout."""

import argparse
from pathlib import Path

UPSTREAM_COMMIT = "4e7fefcc4923f9e083d5324dc75a7d2f906af77f"


def replace_once(text, old, new):
    """Refuse drift or an already patched input instead of silently editing another revision."""
    if text.count(old) != 1:
        raise ValueError(f"expected exactly one upstream context: {old[:80]!r}")
    return text.replace(old, new, 1)


def adapt(root):
    """Gate only 10de:2520 behind its own flag and one exact BDF; add dispatch tests."""
    source = root / "kernel/gpu_unlock/main.c"
    test = root / "tests/test_main.c"
    main = source.read_text()
    main = replace_once(main, "static bool ad103_experimental;", '''/* GA106M Laptop is unvalidated: require explicit, exact-card opt-in. */
static bool ga106m_experimental;
module_param(ga106m_experimental, bool, 0444);
MODULE_PARM_DESC(ga106m_experimental, "permit only experimental 10de:2520 with a single exact PCI address");

static bool ad103_experimental;''')
    main = replace_once(main, "\tunsigned int i;\n\n\tif (pdev->device", "\tunsigned int i;\n\n\tif (pdev->device == 0x2520)\n\t\treturn ga106m_experimental;\n\tif (pdev->device")
    main = replace_once(main, "\twhile ((pdev = next_supported(pdev))) {", '''\twhile ((pdev = next_supported(pdev))) {
\t\tif (pdev->device == 0x2520 &&
\t\t    (all || strcasecmp(targets, pci_name(pdev)))) {
\t\t\tpci_err(pdev, "gpu_unlock: experimental GA106M requires a single exact PCI target\\n");
\t\t\tcontinue;
\t\t}''')
    tests = test.read_text()
    tests = replace_once(tests, "\tad103_experimental = false;", "\tga106m_experimental = false;\n\tad103_experimental = false;")
    tests = replace_once(tests, "\t/* No opt-in on the command line: the module does nothing. */", '''\t/* GA106M: default off, exact target only, no professional IDs or bound GPU. */
\treset("0000:00:10.0");
\tadd("0000:00:10.0", 0x2520, false);
\trun();
\tassert(!fake.loads && !fake.started);
\treset("all");
\tga106m_experimental = true;
\tadd("0000:00:10.0", 0x2520, false);
\trun();
\tassert(!fake.loads && !fake.started);
\tassert(logged("experimental GA106M requires a single exact PCI target"));
\treset("0000:00:10.0,0000:00:11.0");
\tga106m_experimental = true;
\tadd("0000:00:10.0", 0x2520, false);
\trun();
\tassert(!fake.loads && !fake.started);
\treset("0000:00:11.0");
\tga106m_experimental = true;
\tadd("0000:00:10.0", 0x2520, false);
\trun();
\tassert(!fake.loads && !fake.started);
\treset("0000:00:10.0");
\tga106m_experimental = true;
\tadd("0000:00:10.0", RTX_A2000, false);
\trun();
\tassert(!fake.loads && !fake.started);
\treset("0000:00:10.0");
\tga106m_experimental = true;
\tadd("0000:00:10.0", 0x2520, true);
\trun();
\tassert(!fake.started);
\treset("0000:00:10.0");
\tga106m_experimental = true;
\tadd("0000:00:10.0", 0x2520, false);
\tfake.expected_concurrent = 1;
\trun();
\tassert(fake.loads == 1 && fake.started == 1 && fake.unlocks[0]);
\tputs("ga106m: gated exact-card dispatch PASS");

\t/* No opt-in on the command line: the module does nothing. */''')
    # Compute both before writing, so a context failure leaves the checkout intact.
    source.write_text(main)
    test.write_text(tests)


def main():
    """Apply the adaptation to an explicit upstream source directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    adapt(args.root)


if __name__ == "__main__":
    main()
