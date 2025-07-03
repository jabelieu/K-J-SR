import pysr as srp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import sympy as syp

max_eq = 4
min_alpha = 0.2

table_print_flag = 1 # 1 = on , 0 = off

selector = 'score'

nuclei_list = ['ca40','ca44', 'ca48','sn100','sn132','pb208', 'pb266']
# nuclei_list = []
# valid_list = []
valid_list = [ 'ca38' , 'ca42' , 'ca46' , 'ca50' , 'ca54' , 'sn120' ,
              'sn170' , 'ni56' , 'ni68' , 'pb196' ]

if len ( valid_list ) != 0 :
    nuclei_list += valid_list

path_id = '/mnt/home/belieujo/projects/K-J-SR/outputs/pc_7nucl_28_06_2025_16_44_14'

dens_plot_flag = 1 # on = 1 , off = 0
dens_plot_save_flag = 1
plot_save_name = '7nucl_validation_' + selector + '.png'

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

def rmsr ( r , density ) :
    """
    Calculate the root mean square radius from a density profile.
    Parameters
    ----------
    r : array_like
        The radial distances at which the density is defined.
    density : array_like
        The density values at the corresponding radial distances.
    Returns
    -------
    float
        The root mean square radius.
    """

    # for idx,val in enumerate(density) : 
    #     if val < 0 and np.abs(val) < 1e-3 :
    #         density[idx] = 0.0

    # np.clip(density, 0, None, out=density)  # Ensure density is non-negative

    denominator = simpson ( y = r**2 * density , x = r )

    numerator = simpson ( y = r**4 * density , x = r )

    # print('Numerator:', numerator)
    # print('Denominator:', denominator)

    radius = numerator / denominator

    return np.sqrt(radius)

feature_file = 'ca48_j3_quants.txt'
target_file = 'ca48_fort.20'

target_charge_radius40 = 3.493268148170989
target_charge_radius48 = 3.507005257178622
target_charge_radius208 = 5.508300727077247

master_hf_rc_list = [target_charge_radius40,target_charge_radius48,target_charge_radius208]

'''
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
'''

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

for nuclei in nuclei_list :

    feature_file = nuclei + '_j3_quants.txt'
    target_file = nuclei + '_fort.20'

    raw_feature_data = np.loadtxt ( feature_file , skiprows = 1 , dtype = float )

    '''
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
    '''

    r = raw_feature_data [:,0]
    rp = raw_feature_data [:,1]
    rn = raw_feature_data [:,2]
    tp = raw_feature_data [:,3]
    tn = raw_feature_data [:,4]
    ssp = raw_feature_data [:,5]
    ssn = raw_feature_data [:,6]
    sp = raw_feature_data [:,7]
    sn = raw_feature_data [:,8]

    # Append to master arrays
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

ngrid = len ( r )

mfl = np.column_stack ( ( master_rp, master_rn, master_tp, master_tn,
                        master_ssp, master_ssn, master_sp, master_sn ) ) # list of features.

pysr_instance = srp.PySRRegressor()
model = pysr_instance.from_file(run_directory = path_id)

df = model.equations_

eq_sort = df.sort_values(selector)

x_vals = mfl.flatten() if mfl.ndim == 2 else mfl  # make sure X is 1D

colors = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#117733",  # dark green
    "#332288",  # dark blue
    "#88CCEE",  # light blue
    "#44AA99",  # teal
    "#DDCC77",  # sand
    "#AA4499",  # purple
    "#882255",  # wine
    "#661100",  # brown
    "#6699CC",  # steel blue
    "#FFAA00",  # gold
    "#66CCEE",  # light cyan
    "#228833",  # strong green
    "#EE6677",  # pinkish red
]

line_styles = ['--', ':', '-.']  # per equation

fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharex=True)

added_to_legend = [False] * len(nuclei_list) # to avoid legend duplicates
added_to_legend2 = [False] * 30 # to avoid legend duplicates
c = 0

rc_pererr_ls = []
rc_diff_ls = []
pc_diff_ls = []
pcdiff_rmsd_ls = []
pc_custlss_ls = []
eqnumb_list = []
hf_rc_ls = []
sr_rc_ls = []
ncl_ls = []

for i, (idx, row) in enumerate(eq_sort.iterrows()):
    expr = row["lambda_format"]
    expr_loss = row['loss']
    losses = eq_sort['loss'].iloc[:max_eq-1].values
    scaled_loss =  (losses - losses.min()) / (losses.max() - losses.min())
    alphas = 1 - scaled_loss
    alphas = alphas * (1 - min_alpha) + min_alpha
    try :
        y_pred = expr(mfl)
        c += 1
    except :
        continue
    if c == max_eq :
        break
    # Slice predicted and true values
    
    alpha = 1.0 - expr_loss  # fade lower-ranked expressions

    for j in range ( len(nuclei_list) ):

        sidx = j * ngrid
        eidx = (j+1)* ngrid

        axs0_label = nuclei_list[j] if not added_to_legend[j] else None
        axs1_label = str(idx).zfill(2) if not added_to_legend2[idx] else None

        color = colors[j]
        nucleus = nuclei_list [ j ]

        if nucleus in valid_list :
            nucleus += '*'

        pc_sr = y_pred [ sidx : eidx ]
        pc_hf = master_target_data [ sidx : eidx ]

        sr_z = 4*np.pi*simpson ( pc_sr * r ** 2 , r )
        hf_z = 4*np.pi*simpson ( pc_hf * r ** 2 , r )

        N = hf_z / sr_z

        y_pred[sidx:eidx] *= N

        pc_sr = y_pred [ sidx : eidx ]
        pc_hf = master_target_data [ sidx : eidx ]
        pc_scale = np.sum ( pc_sr )

        pc_diff = pc_sr - pc_hf
        scaled_pc_diff = np.abs(pc_diff)/pc_scale

        rmsd_pc_diff = np.sqrt ( np.mean ( pc_diff ** 2 ) )
        pc_custlss = np.sum ( scaled_pc_diff )


        sr_rmsr = rmsr ( r , pc_sr )
        hf_rmsr = rmsr ( r , pc_hf )
        diff_rmsr = sr_rmsr - hf_rmsr
        percerr_rmsr = np.abs(diff_rmsr)/hf_rmsr * 100

        rc_diff_ls.append ( np.abs ( diff_rmsr ) )
        rc_pererr_ls.append ( percerr_rmsr )
        pcdiff_rmsd_ls.append ( rmsd_pc_diff )
        pc_custlss_ls.append ( pc_custlss )
        ncl_ls.append ( nucleus )
        eqnumb_list.append ( str(idx).zfill(2) )
        hf_rc_ls.append ( hf_rmsr )
        sr_rc_ls.append ( sr_rmsr )

        # fz = 4*np.pi*simpson ( pc_sr * r ** 2 , r )

        # print(len(sr_pc),len(hf_pc))

        # Density plot (left)
        if dens_plot_flag == 1 :

            axs[0].plot(
                r, pc_sr,
                linestyle=line_styles[i],
                color=color,
                alpha=alphas[i],
                label=axs0_label,
                zorder=2
            )

            # Residual plot (right)
            axs[1].plot(
                r, pc_diff,
                linestyle=line_styles[i],
                color=color,
                alpha=alphas[i],
                label=axs1_label,
                zorder=2
            )

            added_to_legend[j] = True
            added_to_legend2[idx] = True

if dens_plot_flag == 1 :

    for i in range ( len ( nuclei_list ) ):
        sidx = i * ngrid
        eidx = (i + 1) * ngrid
        axs[0].plot(
            master_r[sidx:eidx],
            master_target_data[sidx:eidx],
            color='black',
            alpha=0.5,
            lw=1,
            zorder=2
        )

    axs[0].set_xlabel("r [ fm ]")
    axs[0].set_ylabel(r"$\rho_c$ [ fm$^{-3}$]")
    axs[0].set_title("Predicted Densities")
    axs[0].legend()
    axs[0].grid(ls='--', alpha=0.3)
    axs[0].tick_params(direction='in')
    axs[0].text ( 0.01, 0.01, "Selection: "+selector, transform=axs[0].transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='left' )

    axs[1].set_xlabel("r [ fm ]")
    axs[1].set_ylabel(r"$\rho_c^p-\rho_c^t$ [ fm$^{-3}$]")
    axs[1].set_title("Residuals")
    axs[1].legend()
    axs[1].grid(ls='--', alpha=0.3)
    axs[1].tick_params(direction='in')

    for ax in axs:
        ax.set_xlim(0, 10)

    if dens_plot_save_flag == 1 :

        plt.savefig(plot_save_name, dpi=300)

if table_print_flag == 1 :

    tp_cols = [ ncl_ls , eqnumb_list , hf_rc_ls , sr_rc_ls ,
    rc_diff_ls , rc_pererr_ls , pcdiff_rmsd_ls , pc_custlss_ls ]
    tp_head = [ 'Nucl.' , 'Eq. #' , 'HF rC' , 'SR rC' , 'rC diff' , '%Err rC' , 'RMSD(pC diff)' , 'pC custLoss' ]
    tp_fmt = [ 's' , 's' , '.3f' , '.3f' ,'.2e','.2f','.2e','.3e' ]

    table_printer(tp_cols,tp_head,tp_fmt)

avg_rc_diff = np.mean ( rc_diff_ls )
rmsd_rc_diff = np.sqrt ( np.mean ( np.array(rc_diff_ls) ** 2) )

avg_pe_rc = np.mean ( percerr_rmsr )
rmsd_pe_rc = np.sqrt ( np.mean ( np.array(percerr_rmsr) ** 2) )

avg_rmsd_pcdiff = np.mean ( pcdiff_rmsd_ls )
rmsd_rmsd_pcdiff = np.sqrt ( np.mean ( np.array(pcdiff_rmsd_ls) ** 2 ) )

avg_pc_custloss = np.mean ( pc_custlss_ls )
rmsd_pc_custloss = np.sqrt ( np.mean ( np.array(pc_custlss_ls) ** 2 ) ) 

print(f'avg rc diff {avg_rc_diff:.3e}')
print(f'rmsd rc diff {rmsd_rc_diff:.3e}')
print(f'avg %err rc {avg_pe_rc:.3e}')
print(f'rmsd %err rc {rmsd_pe_rc:.3e}')
print(f'avg RMSD(pc diff) {avg_rmsd_pcdiff:.3e}')
print(f'rmsd RMSD(pc diff) {rmsd_rmsd_pcdiff:.3e}')
print(f'avg pc custLoss {avg_pc_custloss:.3e}')
print(f'rmsd pc custLoss {rmsd_pc_custloss:.3e}')

# print(model.latex_table(indices=None, precision=3, columns=['equation', 'complexity', 'loss', 'score']))

# def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
#     """Calculate scores for each equation based on loss and complexity.

#     Score is defined as the negated derivative of the log-loss with respect to complexity.
#     A higher score means the equation achieved a much better loss at a slightly higher complexity.
#     """
#     scores = []
#     lastMSE = None
#     lastComplexity = 0

#     for _, row in df.iterrows():
#         curMSE = row["loss"]
#         curComplexity = row["complexity"]

#         if lastMSE is None:
#             cur_score = 0.0
#         else:
#             if curMSE > 0.0:
#                 cur_score = -np.log(curMSE / lastMSE) / (curComplexity - lastComplexity)
#             else:
#                 cur_score = np.inf

#         scores.append(cur_score)
#         lastMSE = curMSE
#         lastComplexity = curComplexity

#     return pd.DataFrame(
#         {
#             "score": np.array(scores),
#         },
#         index=df.index,
#     )