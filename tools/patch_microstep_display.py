"""Display micro-batches (optimizer steps x grad accumulation) in EasyWAM progress.

Display-only patch: the tqdm bar and the '[train] step=x/y' log now count
micro-batches (e.g. 8/33048, same convention as OpenWAM). Optimizer-step
semantics, loss, lr and scheduler are untouched, so it is reusable for the
badminton task as well.

Patched file: official/code/src/trainer.py
  1) train(): total_micro_steps = max_steps * gradient_accumulation_steps
  2) tqdm(total=..., initial=...) uses micro steps
  3) progress_bar.update(1) -> update(micro_steps_per_optimizer_step)
  4) '[train] step=%d/%d' arguments multiplied by grad accumulation
"""
import difflib
import pathlib
import py_compile
import shutil
import time

PATH = pathlib.Path("/datapool/lianjh/easywam/official/code/src/trainer.py")
BACKUP = PATH.with_name(PATH.name + ".bak_microstep_" + time.strftime("%Y%m%d"))

OLD_A = '''        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        progress_bar = tqdm(
            total=self.max_steps,
            initial=self.global_step,'''
NEW_A = '''        logger.info("Starting training with max_steps=%d.", self.max_steps)
        # Display-only: count micro-batches (optimizer steps x gradient accumulation)
        # so the bar reads like OpenWAM (e.g. 8/33048). Optimizer-step semantics and
        # the lr/scheduler schedule are unchanged.
        micro_steps_per_optimizer_step = int(self.gradient_accumulation_steps)
        total_micro_steps = self.max_steps * micro_steps_per_optimizer_step
        data_iter = iter(self.train_loader)
        progress_bar = tqdm(
            total=total_micro_steps,
            initial=self.global_step * micro_steps_per_optimizer_step,'''

OLD_B = "                progress_bar.update(1)"
NEW_B = "                progress_bar.update(micro_steps_per_optimizer_step)"

OLD_C = '''                    ) % (
                        self.epoch,
                        self.global_step,
                        self.max_steps,'''
NEW_C = '''                    ) % (
                        self.epoch,
                        self.global_step * micro_steps_per_optimizer_step,
                        self.max_steps * micro_steps_per_optimizer_step,'''


def main():
    src = PATH.read_text()
    if "micro_steps_per_optimizer_step" in src:
        print("ALREADY_PATCHED")
        return
    for old, new, name in ((OLD_A, NEW_A, "A"), (OLD_B, NEW_B, "B"), (OLD_C, NEW_C, "C")):
        count = src.count(old)
        if count != 1:
            raise SystemExit(f"anchor {name} found {count} times, expected 1")
        src = src.replace(old, new)
    if not BACKUP.exists():
        shutil.copy2(PATH, BACKUP)
    before = PATH.read_text()
    PATH.write_text(src)
    diff = difflib.unified_diff(
        before.splitlines(), src.splitlines(),
        fromfile=str(BACKUP), tofile=str(PATH), lineterm="", n=3)
    print("\n".join(diff))
    py_compile.compile(str(PATH), doraise=True)
    print("PATCH_OK", PATH)
    print("total_micro_steps is derived at runtime: max_steps * gradient_accumulation_steps")


if __name__ == "__main__":
    main()
