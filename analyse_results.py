from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.LogAnalyser import LogAnalyser

FLAGS = flags.FLAGS
flags.DEFINE_string('log_directory', './logs/logs-20-12-18-12-34-53', 'log directory')

def main(argv):
    analyser = LogAnalyser(FLAGS.log_directory)

    gt, pt = analyser.load('points')
    rms = analyser.compare(gt, pt)
    print(rms)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass