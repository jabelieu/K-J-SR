#==============================================================================
# PROGRAM : Results Plotter
# AUTHOR  : Joshua Belieu | Fletch
# DATE    : 20.07.2025
# PURPOSE : Plotting results from PySR runs
#==============================================================================
#                               BEGIN PROGRAM
#------------------------------------------------------------------------------

"""
Script Parameters
"""

density_plot_flag = 1 # 0 = off, 1 = on
nuclear_property_plot_flag = 1
table_print_flag = 1
scaler_flag = 1

run_name = 'wsq_5050_ss_10_07_2025_14_41_23'
path_id = 'outputs/' + run_name
targ_pkl = 'targss_' + run_name + '.pkl'
fl_pkl = 'flss_' + run_name + '.pkl'

nuclei_list = ['ca40','ca44', 'ca48','sn100','sn132','pb208', 'pb266']

valid_list = []
# valid_list = [ 'ca38' , 'ca42' , 'ca46' , 'ca50' , 'ca54' , 'sn120' ,
#               'sn170' , 'ni56' , 'ni68' , 'pb196']#, 'o16' , 'o24' ]

if valid_list != [] :
    nuclei_list += valid_list

properties_to_plot = ['rc_pererr', 'pc_diff_rmsd', 'pc_custlss']
proprty_titles = [
    r'$r_c$ Percent Error',
    r'$(\delta\rho_c)_{RMSD}$',
    r'$\rho_c$ Custom Loss'
]

"""
Imported Libraries
"""

import pysr as srp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
# import sympy as syp
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import pickle
import pandas as pd

"""
Custom Functions
"""

def signature () :

    signature = r'''
    Coded by:
    --------
      ________ ___       _______  _________  ________  ___  ___     
     |\  _____\\  \     |\  ___ \|\___   ___\\   ____\|\  \|\  \    
     \ \  \__/\ \  \    \ \   __/\|___ \  \_\ \  \___|\ \  \\\  \   
      \ \   __\\ \  \    \ \  \_|/__  \ \  \ \ \  \    \ \   __  \  
       \ \  \_| \ \  \____\ \  \_|\ \  \ \  \ \ \  \____\ \  \ \  \ 
        \ \__\   \ \_______\ \_______\  \ \__\ \ \_______\ \__\ \__\
         \|__|    \|_______|\|_______|   \|__|  \|_______|\|__|\|__|
                '''

    print ( signature + '\n' )

    return 0

def table_printer ( column_lists , column_headers , formatters ) :

    num_cols = len ( column_headers )
    num_rows = len ( column_lists [0] )

    str_cols = []
    col_widths = []

    for col , fmt in zip ( column_lists , formatters ) :

        formatted_col = []

        for item in col :

            try :

                formatted_item = format ( item , fmt )

            except TypeError :

                formatted_item = str ( item )

            formatted_col.append ( formatted_item )

        str_cols.append ( formatted_col )

        # Determine max width for the column

        max_data_len = max ( len ( s ) for s in formatted_col )
        header_width = len ( column_headers [ len ( str_cols ) - 1 ] )
        chosen_width = max ( header_width , max_data_len ) + 2
        col_widths.append ( chosen_width )

    # Build header string
    header_str = "|"
    for head , w in zip ( column_headers , col_widths ) :
        header_str += f"{head:^{w}}|"

    # Build separator string
    sep_str = "|" + "|".join ( "-" * w for w in col_widths ) + "|"

    # Build body
    body_str = ""
    for i in range ( num_rows ) :

        row = "|"

        for j in range (num_cols ) :

            row += f"{str_cols[j][i]:^{col_widths[j]}}|"

        body_str += row

        if i != num_rows - 1 :

            body_str += "\n" + sep_str + "\n"

    # Final output
    print("=" * len(header_str))
    print(header_str)
    print(sep_str)
    print(body_str)
    print("=" * len(header_str))

    return 0

def get_contrast_text_color(rgb):
    # rgb: tuple of floats (r, g, b) between 0 and 1
    # Calculate luminance (perceived brightness)
    luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
    return 'black' if luminance > 0.5 else 'white'

def rmsr ( r , density ) :
    """
    Calculate the root mean square radius from a density profile.
    Parameters
    ----------
    r       : Array_like. The radial distances at which the density is defined.
    density : Array_like. The density values at the corresponding radial 
              distances.
    Returns
    -------
    Float. The root mean square radius.
    """

    denominator = simpson ( y = r**2 * density , x = r )

    numerator = simpson ( y = r**4 * density , x = r )

    radius = numerator / denominator

    return np.sqrt(radius)

def load_data ( nuclei_list ) :

    """
    PURPOSE : Provided a list of nuclei, this function loads the feature data
    ------- and target data from the respective files.

    INPUTS
    ------
    nuclei_list : List. A list of nuclei names (strings).

    OUTPUTS
    -------
    master_target_data : Numpy array. The target data for all nuclei.
    mfl : Numpy array. A 2D array containing the features for all nuclei, 
    columnwise.

    Features and target key :

    r : distance in fm
    rp : rho-proton in particle / fm^3
    rn : rho-neutron
    tp : tau-proton in MeV / fm^3
    tn : tau-neutron
    ssp : divergance of spin orbit density-proton
    ssn : divergance of spin orbit density-neutron
    sp : spin orbit density-proton
    sn : spin orbit density-neutron

    rc : rho-charge
    """

    '''
    Initialize master arrays to hold all data from all nuclei.
    '''

    global ngrid
    global r

    master_r = np.array([])
    master_rp = np.array([])
    master_rn = np.array([])
    master_tp = np.array([])
    master_tn = np.array([])
    master_ssp = np.array([])
    master_ssn = np.array([])
    master_sp = np.array([])
    master_sn = np.array([])

    master_target_data = np.array([])

    '''
    Loop through each nucleus in the nuclei_list, loading the feature and target
    data and append them to master arrays.
    '''

    for nuclei in nuclei_list :

        feature_file = nuclei + '_j3_quants.txt'
        target_file = nuclei + '_fort.20'

        raw_feature_data = np.loadtxt ( feature_file , skiprows = 1 , dtype = float )      

        r = raw_feature_data [:,0] 
        rp = raw_feature_data [:,1]
        rn = raw_feature_data [:,2]
        tp = raw_feature_data [:,3]
        tn = raw_feature_data [:,4]
        ssp = raw_feature_data [:,5]
        ssn = raw_feature_data [:,6]
        sp = raw_feature_data [:,7]
        sn = raw_feature_data [:,8]

        master_r = np.append(master_r, r)
        master_rp = np.append(master_rp, rp)
        master_rn = np.append(master_rn, rn)
        master_tp = np.append(master_tp, tp)
        master_tn = np.append(master_tn, tn)
        master_ssp = np.append(master_ssp, ssp)
        master_ssn = np.append(master_ssn, ssn)
        master_sp = np.append(master_sp, sp)
        master_sn = np.append(master_sn, sn)

        with open ( target_file , 'r' ) as f :
            lines = f.readlines()

        target_data_list = []

        for i in range ( 4 , 74 + 1 ) :
            target_data_value = lines [ i ].split() [ 2 ]
            target_data_list.append ( float ( target_data_value ) )

        target_data = np.array ( target_data_list )

        master_target_data = np.append(master_target_data, target_data) 
    
    ngrid = len ( r ) # Number of grid points, used for slicing later.

    '''
    Stack all master arrays columnwise to create a 2D array for features. This 
    is formatted as such for the PySR rergression algorithm.
    '''

    mfl = np.column_stack ( ( master_rp, master_rn, master_tp, master_tn,
                            master_ssp, master_ssn, master_sp, master_sn ) )
    
    return master_target_data , mfl

def get_results ( path_id , nuclei_list ,selector = 'score' , max_eq = 3,) :

    rc_pererr_ls = []
    rc_diff_ls = []
    pc_diff_ls = []
    pcdiff_rmsd_ls = []
    pc_custlss_ls = []
    eqnumb_list = []
    complexity_ls = []
    loss_ls = []
    score_ls = []
    hf_rc_ls = []
    sr_rc_ls = []
    ncl_ls = []
    sr_func_arr_ls = []
    hf_func_arr_ls = []

    target_data , fl = load_data ( nuclei_list )

    if scaler_flag == 1 :

        with open ( targ_pkl , 'rb' ) as f :
            targ_scaler = pickle.load ( f )
        with open ( fl_pkl , 'rb' ) as f :
            fl_scaler = pickle.load ( f )

        unscaled_rp = fl[:,0] # rho-proton

        fl = fl_scaler.transform ( fl )


    pysr_instance = srp.PySRRegressor()
    model = pysr_instance.from_file(run_directory = path_id)

    '''

    df columns : 

    'lambda_format' : The equation in a callable format.
    'sympy_format'  : The equation in sympy format.
    'loss'          : The loss value for the equation.
    'score'         : The score of the equation.
    'complexity'    : The complexity of the equation.
    'equation'      : The equation in string format.

    '''

    df = model.equations_

    eq_sort = df.sort_values(selector)
    if selector == 'score' :
        eq_sort = eq_sort[::-1]
    # print(eq_sort['score'].values[:max_eq])

    for i, (idx, row) in enumerate(eq_sort.iloc[:max_eq].iterrows()):

        expr = row["lambda_format"]
        expr_loss = row['loss']
        expr_score = row['score']
        losses = eq_sort['loss'].iloc[:max_eq].values
        complexity = row['complexity']
        # y_pred = np.array([])
        # print(row['complexity'],row['sympy_format'])
        try :

                y_pred = expr(fl)

                if scaler_flag == 1 :

                    y_pred = targ_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_pred += unscaled_rp # rho-proton

        except :
            continue
    
        for j in range ( len(nuclei_list) ):

            sidx = j * ngrid
            eidx = (j+1)* ngrid

            nucleus = nuclei_list [ j ]

            # if nucleus in valid_list : # TODO make this a parameter.
                # nucleus += '*'

            pc_sr = y_pred [ sidx : eidx ]
            pc_hf = target_data [ sidx : eidx ]

            # sr_z = 4*np.pi*simpson ( pc_sr * r ** 2 , r )
            # hf_z = 4*np.pi*simpson ( pc_hf * r ** 2 , r )

            # N = hf_z / sr_z # TODO make normalization a parameter.

            # y_pred[sidx:eidx] *= N
            # pc_sr *= N


        #     pc_sr = y_pred [ sidx : eidx ]
        #     pc_hf = master_target_data [ sidx : eidx ]
        #     pc_scale = np.sum ( pc_sr )

            pc_diff = pc_sr - pc_hf
        #     scaled_pc_diff = np.abs(pc_diff)/pc_scale

            rmsd_pc_diff = np.sqrt ( np.mean ( pc_diff ** 2 ) )
            pc_custlss = np.sum ( pc_diff )

            sr_rmsr = rmsr ( r , pc_sr )
            hf_rmsr = rmsr ( r , pc_hf )
            diff_rmsr = sr_rmsr - hf_rmsr
        #     # print('cheese')
            percerr_rmsr = np.abs(diff_rmsr)/hf_rmsr * 100

            rc_diff_ls.append ( np.abs ( diff_rmsr ) )
            rc_pererr_ls.append ( percerr_rmsr )
            pcdiff_rmsd_ls.append ( rmsd_pc_diff )
            pc_custlss_ls.append ( pc_custlss )
            ncl_ls.append ( nucleus )
            sr_func_arr_ls.append ( pc_sr )
            hf_func_arr_ls.append ( pc_hf )
            loss_ls.append ( expr_loss )
            pc_diff_ls.append ( pc_diff )
            score_ls.append ( expr_score )
            complexity_ls.append ( complexity )
        #     eqnumb_list.append ( str(idx).zfill(2) )
            hf_rc_ls.append ( hf_rmsr )
            sr_rc_ls.append ( sr_rmsr )

        res_df = pd.DataFrame({
            'nucleus' : ncl_ls,
            'hf_rmsr' : hf_rc_ls,
            'sr_rmsr' : sr_rc_ls,
            'rc_diff' : rc_diff_ls,
            'rc_pererr' : rc_pererr_ls,
            'pc_diff_rmsd' : pcdiff_rmsd_ls,
            'pc_custlss' : pc_custlss_ls,
            'complexity' : complexity_ls,
            'sr_func_arr' : sr_func_arr_ls,
            'hf_func_arr' : hf_func_arr_ls,
            'pc_diff' : pc_diff_ls,
            'loss' : loss_ls,
            'score' : score_ls
        })

    return res_df

def density_plot ( df , save_name = 'density_plot_bar' , colormap_name = 'viridis' ) :

    unique_nuclei = df['nucleus'].unique()
    cmap =  plt.get_cmap(colormap_name).resampled(len(unique_nuclei))
    color_dict = {nuc: cmap(i) for i, nuc in enumerate(unique_nuclei)}

    loss_values = df['loss'].values
    min_loss, max_loss = loss_values.min(), loss_values.max()
    loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

    min_alpha, max_alpha = 0.3, 1.0
    loss_alpha = lambda loss: max_alpha- (loss - min_loss) / loss_range * ( max_alpha - min_alpha )

    seen_labels = set()
    seen_complexities = set()

    unique_complexities = sorted(df['complexity'].unique())
    line_styles = ['--', ':', '-.']
    style_dict = {
        comp: line_styles[i % len(line_styles)]
        for i, comp in enumerate(unique_complexities)
    }

    fig, axs = plt.subplots ( 1 , 2 , figsize = ( 12 , 6 ) )

    for i , row in df.iterrows():

        nucleus = row['nucleus']
        color = color_dict[nucleus]
        loss = row['loss']
        score = row['score']
        complexity = row['complexity']

        label = nucleus if nucleus not in seen_labels else None
        seen_labels.add(nucleus)
        label2 = f'Complexity {complexity}' if complexity not in seen_complexities else None
        seen_complexities.add(complexity)
        linestyle = style_dict[complexity]
        alpha = loss_alpha(loss)
        ax1, ax2 = axs

        ax1.plot(r, row['sr_func_arr'],color=color,label=label,linestyle=linestyle,alpha=alpha)
        ax1.set_xlabel('r [ fm ]')
        ax1.set_ylabel(r'$\rho$ [ fm$^-3$ ] ')

        ax2.plot(r, row['pc_diff'],color=color,linestyle=linestyle,label=label2,alpha=alpha)
        ax2.set_xlabel('r [ fm ]')
        ax2.set_ylabel(r'$\rho$ [ fm$^-3$ ] ')

    for nucleus in df['nucleus'].unique():
        hf_row = df[df['nucleus'] == nucleus].iloc[0]  # just one example per nucleus
        ax1.plot(
        r,
        hf_row['hf_func_arr'],
        color='black',
        linestyle='-',
        linewidth=1.0,     # thinner line
        alpha=0.4,         # faint presence
        zorder=1           # drawn underneath everything else
    )


    for ax in axs:

        handles, labels = ax.get_legend_handles_labels()
        for h in handles:
            h.set_alpha(1.0)  # override transparency in legend

        ax.legend()
        ax.grid(ls='--', alpha=0.5)
        ax.set_xlim(0, 10)

    plt.tight_layout()
    plt.savefig( 'plots/' + save_name + '.png' , dpi = 300 )
    
    return 0

def nuclear_property_plot(
    df,
    property_col,
    complexity_col='complexity',
    nucleus_col='nucleus',
    property_title=None,
    save_name='property_by_complexity_grid',
    cmap_name='viridis'
):
    """
    Plot a 1x3 grid of nuclei colored by a property, split by complexity values.

    Parameters:
    - df : DataFrame with nucleus data
    - property_col : str, name of the property column to color by
    - complexity_col : str, name of the complexity column (used for splitting into subplots)
    - nucleus_col : str, name of the nucleus name column
    - valid_list : list of nuclei to mark with '*', optional
    - property_title : str, title for the entire figure
    - save_name : str, filename for saving the plot
    - cmap_name : str, matplotlib colormap name
    """

    # Get unique complexities and sort (expecting 3)
    complexities = sorted(df[complexity_col].unique())
    if len(complexities) != 3:
        print(f"Warning: Expected exactly 3 unique complexities, found {len(complexities)}")

    # Prepare subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))#, constrained_layout=True)

    cmap = plt.colormaps.get_cmap(cmap_name)

    # Get global norm for the property over whole df
    prop_values_all = df[property_col].values
    norm = Normalize(vmin=np.nanmin(prop_values_all), vmax=np.nanmax(prop_values_all))

    for ax, comp in zip(axs, complexities):
        sub_df = df[df[complexity_col] == comp]
        nuclei = sub_df[nucleus_col].values
        values = sub_df[property_col].values

        num_nuclei = len(nuclei)
        cols = int(np.ceil(np.sqrt(num_nuclei)))
        rows = int(np.ceil(num_nuclei / cols))

        # Draw grid squares
        for idx, (name, value) in enumerate(zip(nuclei, values)):
            if name in valid_list:
                name_display = name + '*'
            else:
                name_display = name

            row = idx // cols
            col = idx % cols
            color = cmap(norm(value))

            rect = plt.Rectangle((col, rows - row - 1), 1, 1,
                                 facecolor=color, edgecolor='black')
            ax.add_patch(rect)

            text_color = get_contrast_text_color(color)

            ax.text(
                col + 0.5, rows - row - 0.5, name_display,
                ha='center', va='center', fontsize=12,
                color=text_color
            )

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.set_title(f'Complexity {comp}')

        # Colorbar for this subplot
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)

    if property_title:
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        fig.suptitle(property_title, fontsize=16, y=0.95)

    plt.savefig('plots/' + save_name + '.png', dpi=300)

    return 0

"""
Main Execution
"""

signature()

x = get_results(path_id, nuclei_list)

if density_plot_flag == 1 :

    density_plot(x)

if nuclear_property_plot_flag == 1 :

    for prop,title in zip(properties_to_plot, proprty_titles):
        nuclear_property_plot(
            df=x,
            property_col=prop,
            complexity_col='complexity',
            nucleus_col='nucleus',
            property_title=title,
            save_name=prop + '_grid',
            cmap_name='viridis'
        )

if table_print_flag == 1 :

    tp_cols =  [
        'nucleus', 'complexity','score','hf_rmsr','sr_rmsr',
        'rc_diff', 'rc_pererr', 'pc_diff_rmsd', 'pc_custlss'
    ]

    tp_head = [
        'Nucl.', 'Compl.','Score', 'HF rC', 'SR rC',
        'rC diff', '%Err rC', 'RMSD(pC diff)', 'pC custLoss'
    ]

    tp_fmt = [
    's', 'd','.3e' ,'.3f', '.3f',
    '.3e', '.2e','.3e', '.3e'    
    ]

    tp_data = [x[col].tolist() for col in tp_cols]

    table_printer(tp_data, tp_head, tp_fmt)

'''
'nucleus' : ncl_ls,
'hf_rmsr' : hf_rc_ls,
'sr_rmsr' : sr_rc_ls,
'rc_diff' : rc_diff_ls,
'rc_pererr' : rc_pererr_ls,
'pc_diff_rmsd' : pcdiff_rmsd_ls,
'pc_custlss' : pc_custlss_ls,
'complexity' : complexity_ls,
'sr_func_arr' : sr_func_arr_ls,
'hf_func_arr' : hf_func_arr_ls,
'pc_diff' : pc_diff_ls,
'loss' : loss_ls,
'score' : score_ls
'''

#------------------------------------------------------------------------------
#                               END PROGRAM
#==============================================================================