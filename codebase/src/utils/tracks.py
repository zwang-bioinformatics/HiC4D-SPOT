# Author: Bishal Shrestha
# Date: 03-24-2025  

import os

# A function that puts the HiC and tad for all timepoints in a single file with a loop like below hic_track.ini file and then save it to the file:
def hic_track(ids, chr, region, dir_out, output_filename):
    print("Plotting HiC track")
    start = region[1]
    end = region[2]
    for data_label in ["true", "true_perturbed", "pred", "anomaly"]:
        # Create a file to save the HiC track
        file_output = dir_out+"/tracks"
        os.makedirs(file_output, exist_ok=True)
        file = open(file_output+f"/{output_filename}_hictrack_{data_label}.ini", "w")
        
        # Loop through all time points
        file.write("[spacer]\n\n")
        file.write("[x-axis]\nwhere = top\nfontsize=20\n\n")
        file.write("[scale_bar]\nfile_type = scalebar\nfontsize=20\n\n")
        for idx, timePoint in enumerate(ids):

            file.write("[spacer]\n\n")
            
            # HiC matrix
            file.write(f"[hic matrix]\nfile = {dir_out}/.cool/{output_filename}_{data_label}_t{idx+1}.cool\ntitle = T{idx+1}\ndepth = 2000000\nfile_type = hic_matrix\nmin_value=0\nmax_value=1\ncolormap=['white', 'red']\nfontsize=20\n\n")
            
            # Insulation score
            # check if the file exists, if yes, continue
            bedgraph_matrix_file = f'{dir_out}/hicFindTADs/{output_filename}_{data_label}_t{idx+1}_tad_score.bm'
            if os.path.exists(bedgraph_matrix_file):
                file.write(f"[bedgraph_matrix]\nfile = {bedgraph_matrix_file}\nheight = 1.5\ntype = lines\nshow_data_range = false\nfile_type = bedgraph_matrix\n\n")
            
            
        file.close()
        print(f"HiC track file saved in {output_filename}_hictrack_{data_label}.ini")
    
        command = f'hicPlotTADs --tracks {file_output}/{output_filename}_hictrack_{data_label}.ini -o {file_output}/{output_filename}_hictrack_{data_label}_{start}_{end}.jpg --region {chr}:{start}-{end}' # --dpi 350
        os.system(command)
    

def combine_track(region, dir_out, output_filename):
    import os
    from PIL import Image
    import matplotlib.pyplot as plt
    dir_out_org = dir_out
    dir_out = dir_out+"/tracks"
    
    start = region[1]
    end = region[2]
    
    # Load the images
    true_img = Image.open(f"{dir_out}/{output_filename}_hictrack_true_{start}_{end}.jpg")
    true_perturbed_img = Image.open(f"{dir_out}/{output_filename}_hictrack_true_perturbed_{start}_{end}.jpg")
    # pred_img = Image.open(f"{dir_out}/{output_filename}_hictrack_pred_{start}_{end}.jpg")
    anomaly_img = Image.open(f"{dir_out}/{output_filename}_hictrack_anomaly_{start}_{end}.jpg")
    
    # Combine the images
    combined_img = Image.new('RGB', (true_img.width * 3, true_img.height))
    combined_img.paste(true_img, (0, 0))
    combined_img.paste(true_perturbed_img, (true_img.width, 0))
    # combined_img.paste(pred_img, (true_img.width * 2, 0))
    combined_img.paste(anomaly_img, (true_img.width * 2, 0))
    
    # Save the combined image with matplotlib to set the DPI
    fig = plt.figure(figsize=(combined_img.width/100, combined_img.height/100), dpi=400)  # Increase DPI to 400
    ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.imshow(combined_img)
    plt.savefig(f"{dir_out_org}/{output_filename}_hictrack_combined_{start}_{end}.jpg", dpi=400)  # Increase DPI to 400
    plt.close()
    