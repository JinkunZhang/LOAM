import matplotlib
matplotlib.use('Agg')
import os
import numpy as np
import matplotlib.pyplot as plt
import colormaps as cmaps
import time

if __name__ == '__main__':
    case_codes = ['ER','grid','tree','fog','geant','LHC','Dtelekom','SW']
    print_case = ['ER','grid','Tree','Fog','GEANT','LHC','DTelekom','SW']
    case_print_names = dict(zip( case_codes,  print_case))
    group_codes = ['A']
    bar_codes = ['1','2','3','4','5','6']
    cache_print_names = {
        ('A','1') : 'CloudEC', ('A','2') : 'EdgeEC',
        ('A','3') : 'SEPLFU', ('A','4') : 'SEPACN',
        ('A','5') : 'LOAM-GCFW', ('A','6') : 'LOAM-GP',
        }
    #cache_types = ['SGP','SPOO','LCOR','LPR']
    #print_cache = ['SGP','SPOO','LCOR','LPR']#,'BP']
    #cache_print_names = dict(zip( cache_types,  print_cache))
    #hatch_types = dict(zip(cache_types,['-','/','\\','//','\\\\']))
    group_hatch_types = dict(zip(group_codes,['/','\\','x']))
    bar_hatch_types = dict(zip(bar_codes,['/','\\','//','\\\\','x','-']))
    
    # Read and compute average for each file
    case_avg = {}       # dict { 'filename' : average }
    for case in case_codes:
        case_path = os.getcwd() + "/Results/" + case
        files = os.listdir(case_path)
        for f_name in files:
            f = open(case_path+"/"+f_name,'r')
            case_results = []
            for line in f:
                #print("line="+str(line))
                case_results.append(float(line))
            #print("case_res="+str(case_res))
            case_avg[f_name] = np.mean(np.array(case_results))
            #print("case_avg="+str(case_avg))
        
    
    # Sort averages and normalize 
    hatch_height = {}   # { case: group: bar: origin_height }
    for case_code in case_codes:
        hatch_height[case_code] = {}
        for group_code in group_codes:
            hatch_height[case_code][group_code] = {}
    hatch_height_max = {}   # {case: max_height}
    for case_code in case_codes:
        hatch_height_max[case_code] = 0.0
    
            
    for f_name in case_avg:
        # detect case_code and group_code and bar_code, and save the max of each case_code for normalization
        for i in case_codes:
            if f_name.startswith(i):
                case_code = i
                for j in group_codes:
                    if f_name.startswith(case_code+'_'+j):
                        group_code = j
                        for k in bar_codes:
                            if f_name.startswith(case_code+'_'+group_code+'_'+k):
                                bar_code = k
        
        # save
        hatch_height[case_code][group_code][bar_code] = case_avg[f_name]
        if case_avg[f_name] > hatch_height_max[case_code]:
            hatch_height_max[case_code] = case_avg[f_name]

    hatch_height_norm = hatch_height

    # normalize
    for case_code in case_codes:
        for group_code in group_codes:
            for bar_code in bar_codes:
                hatch_height_norm[case_code][group_code][bar_code] /= \
                    ( hatch_height_max[case_code] if hatch_height_max[case_code] > 0.01 else 1.0)
   
    # plot parameters
    #line_num = 1
    #case_line_num = {0: ['ER','grid','tree','fog'], 1:['greant','LHC','Dtelekom','SW']}  # on which line is each case
    plot_case = case_codes #['ER','grid','tree','fog']      # if two lines, plot one at a time    
    figHeight = 1.7                         # height of single line
    bar_width = 0.27     # width of a single bar
    group_space = 0.5 * bar_width   # space between two groups in a singe case
    case_space = 1.5 * bar_width    # space between two cases
    marg_width = bar_width/1.5                # width of margin at left and right to the bars
    init_pos = marg_width + bar_width/2 # x axis of the center of first bar
    is_grid = False                      # wheter print grid
    is_legend = True                  # whether the plot contains legend
    is_twoLines = False                  # whether plot a second line
    subplot_distance = figHeight * 0.2  # distance between two line of figures, not used in single line

    # plot
    #cases = len(plot_case)
    #caches = np.amax(np.array([ len(hatch_height_norm[key]) for key in hatch_height_norm ]))

    # create figure
    group_width = len(bar_codes) * bar_width
    case_width = len(group_codes) * group_width + (len(group_codes)-1) * group_space
    figWidth = len(plot_case) * case_width + (len(plot_case)-1) * case_space
    fig = plt.figure(figsize=(figWidth, figHeight * 2 + subplot_distance))

    # first line
    ax = fig.add_subplot(2, 1, 1)

    # plot bars
    #bar_names = {}  # dict {bar: legend}
    bar_rects = {}
    case_name_pos = {}  # dict {case name: pos}
    xticks = []     # position of ticks x axis
    xtick_lab = []  # print name of cases, for xticks
    max_bar_pos = 0.0
    for case in plot_case:
        case_id = plot_case.index(case)     # pos of case
        for group in group_codes:
            group_id = group_codes.index(group)     # pos of case
            for bar in bar_codes:
                bar_id = bar_codes.index(bar)      # pos of bar
                bar_tuple = (case,group,bar)
                # detailed bar location, legend
                bar_color_factor = 0.9 - 0.65* ((group_id * len(bar_codes) + bar_id) / (len(group_codes)* len(bar_codes)))
                bar_pos = init_pos + case_id* (case_width + case_space) + group_id* (group_width + group_space) + bar_id* bar_width
                bar_name = cache_print_names[(group,bar)]
                bar_rects[bar_tuple] = \
                        ax.bar(bar_pos,
                        hatch_height_norm[case][group][bar],
                        bar_width,
                        #color=cmaps.magma(0.3 + 0.7 * bar_id / len(bar_codes)),
                        color=cmaps.magma( bar_color_factor ),
                        #hatch = group_hatch_types[group],
                        hatch = bar_hatch_types[bar],
                        edgecolor = "black",
                        linewidth=1,
                        #label=bar_name if not labeled else None
                        )
                #bar_names[bar] = bar_name
                if bar_pos > max_bar_pos:
                    max_bar_pos = bar_pos
        
        # position of case name on x axis
        case_name_pos[case] = init_pos + case_width/2 + case_id * (case_width + case_space)
        xticks.append(case_name_pos[case])
        xtick_lab.append(case_print_names[case])

    # adjust figure
    xlim_min = 0.0
    xlim_max = max_bar_pos + bar_width/2 + marg_width
    ax.set_xlim([xlim_min, xlim_max])   # range of x axis
    ax.set_ylim([0,1])                  # range of y axis
    ax.set_xticks(xticks)                           # put on case names
    ax.set_xticklabels(xtick_lab, fontsize = 17 )
    ax.tick_params(axis='y', labelsize= 15)
    ax.grid(is_grid)

    # second line 
    if is_twoLines:
        bx = fig.add_subplot(2, 1, 2)
        plt.tight_layout()
        plot_case_b = ['geant','LHC','Dtelekom','SW']
        bar_rects_b = {}
        case_name_pos_b = {}  # dict {case name: pos}
        xticks = []     # position of ticks x axis
        xtick_lab = []  # print name of cases, for xticks
        max_bar_pos = 0.0
        for case in plot_case_b:
            case_id = plot_case_b.index(case)     # pos of case
            for group in group_codes:
                group_id = group_codes.index(group)     # pos of case
                for bar in bar_codes:
                    bar_id = bar_codes.index(bar)      # pos of bar
                    bar_tuple = (case,group,bar)
                    # detailed bar location, legend
                    bar_color_factor = 0.9 - 0.65* ((group_id * len(bar_codes) + bar_id) / (len(group_codes)* len(bar_codes)))
                    bar_pos = init_pos + case_id* (case_width + case_space) + group_id* (group_width + group_space) + bar_id* bar_width
                    bar_name = cache_print_names[(group,bar)]
                    bar_rects_b[bar_tuple] = \
                            bx.bar(bar_pos,
                            hatch_height_norm[case][group][bar],
                            bar_width,
                            #color=cmaps.magma(0.3 + 0.7 * bar_id / len(bar_codes)),
                            color=cmaps.magma( bar_color_factor ),
                            #hatch = group_hatch_types[group],
                            hatch = bar_hatch_types[bar],
                            edgecolor = "black",
                            linewidth=1,
                            #label=bar_name if not labeled else None
                            )
                    #bar_names[bar] = bar_name
                    if bar_pos > max_bar_pos:
                        max_bar_pos = bar_pos
            
            # position of case name on x axis
            case_name_pos[case] = init_pos + case_width/2 + case_id * (case_width + case_space)
            xticks.append(case_name_pos[case])
            xtick_lab.append(case_print_names[case])

        # adjust figure
        xlim_min = 0.0
        xlim_max = max_bar_pos + bar_width/2 + marg_width
        bx.set_xlim([xlim_min, xlim_max])   # range of x axis
        bx.set_ylim([0,1])                  # range of y axis
        bx.set_xticks(xticks)                           # put on case names
        bx.set_xticklabels(xtick_lab, fontsize = 17 )
        bx.tick_params(axis='y', labelsize= 15)
        bx.grid(is_grid)


    # set legends
    if is_legend:
        # which bars should be included in the legend
        legend_tuples = [     
            ('ER','A','1'),  ('ER','A','2'),  ('ER','A','3'),  ('ER','A','4'), 
            ('ER','A','5'),  ('ER','A','6')
            ]
        # corresponding rects and legend texts
        #print(bar_rects.keys())
        legend_rects = [bar_rects[tup] for tup in legend_tuples]
        legend_texts = [cache_print_names[ (tup[1],tup[2]) ] for tup in legend_tuples]

        # add legend
        lgd = ax.legend(legend_rects,
                            legend_texts,
                            prop={'size': 16},                      # legend font size
                            loc='lower left',                       # where does distance start count (LB corner)
                            bbox_to_anchor=(0.07, 1.08, 0.86, 0.2),       # (x distance to 'loc', y distance to 'loc', length of box, hight of box), 
                                                                    # all mesured by the figure size (width or height)
                            mode='expand',
                            ncol= 6,              # how mant columns in legend
                            borderaxespad=0.)
        ax.grid(zorder=0)


    # plot path
    localtime = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()) 
    if is_twoLines:
        bar_outfile = 'BarFig_('+('-'.join(plot_case))+'_'+('-'.join(plot_case_b))+')_'+localtime
    else:
        bar_outfile = 'BarFig_('+('-'.join(plot_case))+')_'+localtime


    if is_legend:
        fig.savefig(bar_outfile+'.pdf'
                        ,bbox_extra_artists=(lgd, )
                        ,bbox_inches='tight'
                        )
    else:
        fig.savefig(bar_outfile+'.pdf'
                        #,bbox_extra_artists=(lgd, )
                        ,bbox_inches='tight'
                    )

    plt.close(fig)
    exit()
