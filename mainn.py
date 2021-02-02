import utils as utils
import matplotlib.pyplot as plt


no_fight_path = "data/Peliculas/noFights/"
fight_path = "data/Peliculas/fights/"
path_end_avi = ".avi"
path_end_mpg = ".mpg"
path_end_mp4 = ".mp4"



draw_graphs = True
max_peak_violent, max_peak_non_violent, num_peaks_non_violent, num_peaks_violent = utils.run_and_draw_graph_of_speed_and_acc_Pelicus(no_fight_path, fight_path, path_end_mp4, path_end_mp4, draw_graphs)
#utils.Peliculas_results(max_peak_violent, max_peak_non_violent, num_peaks_non_violent, num_peaks_violent)

#path = fight_path+str(38)+path_end_avi
#path_2 = no_fight_path+str(2)+path_end_mpg
#utils.draw_borders_on_vid(path)


