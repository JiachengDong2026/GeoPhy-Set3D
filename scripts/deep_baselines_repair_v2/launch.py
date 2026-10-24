"""Inference/diagnostic/pilot/formal launcher using established UUID locks and double snapshots."""
from repair_context import *
import argparse
import controller


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--key',required=True);ap.add_argument('command',nargs=argparse.REMAINDER)
    a=ap.parse_args();ROOT.mkdir(exist_ok=True)
    original=c.scheduler
    def restricted():
        s=original();devices=s.devices
        s.devices=lambda snapshot:[d for d in devices(snapshot) if d['uuid']==UUID]
        return s
    c.scheduler=restricted;controller.ROOT=ROOT
    cmd=a.command[1:] if a.command and a.command[0]=='--' else a.command
    controller.run(a.key,cmd)


if __name__=='__main__':main()
