import sys
from analysis.plot_utils import disp_reg_hist, disp_reg_hist_smoothed

def main():
  if len(sys.argv) > 1:
    # example arg: ./outputs/20-07-49/outputs
    _OUTPUT_PATH = sys.argv[1]
    disp_reg_hist_smoothed(_OUTPUT_PATH, show=False)
  else: 
    print("Missing output path")

if __name__ == "__main__":
  main()