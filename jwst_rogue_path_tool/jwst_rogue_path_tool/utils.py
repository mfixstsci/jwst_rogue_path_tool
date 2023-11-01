import json
import os

from astroquery.simbad import Simbad


__location__ = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def get_config():
    """Return a dictionary that holds the contents of the ``jwql``
    config file.

    Returns
    -------
    settings : dict
        A dictionary that holds the contents of the config file.
    """
    config_file_location = os.path.join(__location__, 'jwst_rogue_path_tool', 'config.json')

    # Make sure the file exists
    if not os.path.isfile(config_file_location):
        raise FileNotFoundError('The jwst_rogue_path_tool requires a configuration file '
                                'to be placed within the main directory. '
                                'This file is missing.')

    with open(config_file_location, 'r') as config_file_object:
        try:
            # Load it with JSON
            settings = json.load(config_file_object)
        except json.JSONDecodeError as e:
            # Raise a more helpful error if there is a formatting problem
            raise ValueError('Incorrectly formatted config.json file. '
                             'Please fix JSON formatting: {}'.format(e))

    return settings

def querysimbad(ra, dec, rad=1, band='K', maxmag=6., simbad_timeout=200):
    """ Function to put together a "query by criteria" SIMBAD query 
    and return an astropy Table with the results.
    Query criteria here are a circle radius and a faint magnitude limit
    based on a user-selectable bandpass
    """

    Simbad.TIMEOUT = simbad_timeout
    Simbad.reset_votable_fields()
    
    for filtername in ['J','H','K']:
        for prop in ['','_bibcode','_error','_name','_qual','_system','_unit','data']:
            field = 'flux{}({})'.format(prop,filtername)
            Simbad.add_votable_fields(field)

    if ra >=0.:
        ra_symbol = '+'
    else:
        ra_symbol = '-'

    if dec >=0.:
        dec_symbol = '+'
    else:
        dec_symbol = '-'
    
    crit = 'region(circle, ICRS, {}{} {}{},{}d) & ({}mag < {})'.format(ras, ra, decs, dec, rad, band, maxmag)
    print(crit)
    t = Simbad.query_criteria(crit)
    return t

def get_pointing_info(header=None, visit_id=None):
    """Function to obtain the (v1_ra,v1_dec,v3_pa) of a visit.
       If a header is passed, it will get the info from it, if the visit_id is passed, 
       it will query the PPSDB visit_execution table to obtain the same info
    """
    if (header is not None) and (visit_id is not None):
        print('Cannot specify both a header and a visit_id')
        assert False

    if header is not None:
        return header['RA_V1'],header['DEC_V1'],header['PA_V3']

    if visit_id is not None:
        pass
        # Need to write ppsdb query but need to set up the env correctly firs

def DN_report(self,attitudes,RP_padding=0.,draw_reports=True,background_params=[{'threshold':0.1,'func':np.min}],
                save_report_dir=None,save_figures_dir=None,verbose=False,smooth=None):
    '''
    Method to call fixed_angle multiple times and get an estimated DN/pix/s for each
    filter in this observation, as a function of v3PA

    Paramters
    ---------

    attitudes: numpy array
        values of the attitudes at which one wants to compute the DN

    background_params: list of dictionarie (can be None)
        the claw flux is compared to threshold*func(background)
        where func is np.mean/np.min/np.max/np.median 
        (or other callable that returns some summary stats),
        for each of the items of the list

    '''
    
    tmA = []
    tmB = []
    tcA = []
    tcB = []

    for i,att in enumerate(attitudes):
        rd = self.fixed_angle(att,RP_padding=RP_padding,draw_allexp=False,draw_summary=False,smooth=smooth)
        if i == 0:
            tot_exp_dur = rd['tot_exp_dur']
            filtershort_all = rd['filtershort_all']
            filternames = rd['filternames']
            pupilnames = rd['pupilnames']

        tmA.append(rd['totmag_A']) 
        tmB.append(rd['totmag_B']) 
        tcA.append(rd['totcts_A']) 
        tcB.append(rd['totcts_B']) 

    tmA = np.array(tmA)
    tmB = np.array(tmB)
    tcA = np.array(tcA)
    tcB = np.array(tcB)

    if (self.modules == 'ALL') | (self.modules == 'BOTH'):
        tms = [tmA,tmB]
        tcs = [tcA,tcB]
    else:
        if self.modules[0] == 'A':
            tms = [tmA]
            tcs = [tcA]
        elif self.modules[0] == 'B':
            tms = [tmB]
            tcs = [tcB]

    if draw_reports == True:

        nexpspec = np.max(np.array([tcA.shape[1],tcB.shape[1]]))
        nmodules = len(self.SRlist)
        
        f,ax = plt.subplots(3,nmodules,figsize=(5*nmodules,6),sharex=True,sharey='row',squeeze=False)

        j_m = np.empty([attitudes.size,nmodules])
        h_m = np.empty([attitudes.size,nmodules])
        k_m = np.empty([attitudes.size,nmodules])
        DNs = np.empty([attitudes.size,nexpspec,nmodules])
        for i,att in enumerate(attitudes):
            for j in range(nmodules):
                j_m[i,j] = tms[j][i]['j_m']
                h_m[i,j] = tms[j][i]['h_m']
                k_m[i,j] = tms[j][i]['k_m']
                DNs[i,:,j] = tcs[j][i]

        for j in range(nmodules):
            ax[0,j].plot(attitudes,j_m[:,j])
            ax[1,j].plot(attitudes,h_m[:,j])
            ax[2,j].plot(attitudes,k_m[:,j])

    
        ax[0,0].set_ylabel('j_m')
        ax[1,0].set_ylabel('h_m')
        ax[2,0].set_ylabel('k_m')
        
        for k in range(3):
            ax[k,0].set_ylim(24,12)

        ax[0,0].set_xlim(0,360)
        for j in range(nmodules):
            ax[2,j].set_xlabel('V3_PA')
            ax[0,j].set_title('Module {}'.format(self.SRnames[j]))
            

        f.tight_layout()
        
        if save_figures_dir is not None:
            f.savefig(save_figures_dir+'PID{}_obsid{}_mag_sweep.pdf'.format(self.program,self.obsid))

        nexpspec = np.max(np.array([tcA.shape[1],tcB.shape[1]]))
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']

        if background_params is not None:
            fi = filter_info()
            zpc = zero_point_calc()
            bg = []
            ra = self.target_ra
            dec = self.target_dec
            for j in range(nexpspec):
                wave = fi.get_info(pupilnames[j],filternames[j])
                PHOTMJSR = zpc.get_avg_quantity(pupilnames[j],filternames[j],quantity='PHOTMJSR')
                bg.append(jbt.background(ra,dec,wave))
            
            res = []
            
            for bp in background_params:    
                below_threshold = np.ones_like(DNs,dtype=np.bool_)
                    
                for k in range(nexpspec):
                    bck_lambda = bp['threshold']*bp['func'](bg[k].bathtub['total_thiswave'])/PHOTMJSR*1000.
                    for j in range(nmodules):
                        below_threshold[:,k,j] = DNs[:,k,j]*1000./tot_exp_dur[k] < bck_lambda

                below_threshold = np.all(below_threshold,axis=(1,2))
                V3PA_validranges_obs_starts= []
                V3PA_validranges_obs_ends  = []

                change = np.where(below_threshold[:-1] != below_threshold[1:])[0]
                if change.size >0:
                    if below_threshold[change[0]]:
                        change = np.roll(change,1)
                
                    V3PA_validranges_obs_starts = attitudes[change[::2]]
                    V3PA_validranges_obs_ends = attitudes[change[1::2]]
                else:
                    V3PA_validranges_obs_starts = None
                    V3PA_validranges_obs_ends = None
                if verbose == True:
                    print('{:3.1f} x {}(bkg)'.format(bp['threshold'],bp['func'].__name__))
                    for s,e in zip(V3PA_validranges_obs_starts,V3PA_validranges_obs_ends):
                        print(s,e)
                    
                res.append({'s':V3PA_validranges_obs_starts,'e':V3PA_validranges_obs_ends,'bt':below_threshold})
                
        if save_report_dir is not None:
            for r,bp in zip(res,background_params):
                filenm = 'PID{}_obsid{}_report_thr{}_{}.txt'.format(self.program,self.obsid,bp['threshold'],bp['func'].__name__)
                with open(save_report_dir+filenm, 'w') as the_file:
                    the_file.write('*** Valid ranges for PID: {}, obsid:{} ****\n'.format(self.program,self.obsid))
                    if r['s'] is not None:
                        for s,e in zip(r['s'],r['e']):
                            the_file.write('PA Start -- PA End: {} -- {}\n'.format(s,e))
                    else:
                        the_file.write('PA Start -- PA End: {} -- {}\n'.format(0.,360.))

        f,ax = plt.subplots(nexpspec,nmodules,figsize=(5*nmodules,nexpspec*2),sharex=True,sharey=True,squeeze=False)

        for k in range(nexpspec):
            for j in range(nmodules):
                ax[k,j].plot(attitudes,DNs[:,k,j]*1000./tot_exp_dur[k])
                if background_params is None:
                    ax[k,j].axhline(1,linestyle='dashed',label='1DN/pix/ks',c=colors[1])

                else:
                    for l,bp in enumerate(background_params):
                        bck_lambda = bp['threshold']*bp['func'](bg[k].bathtub['total_thiswave'])/PHOTMJSR*1000.
                        ax[k,j].axhline(bck_lambda,linestyle='dashed',label='{:3.1f} x {}(bkg) = {:5.1f} DN/pix/ks'.format(bp['threshold'],bp['func'].__name__,bck_lambda),c=colors[1+l])                        
                        x2p = np.copy(attitudes)
                        y2p = np.copy(DNs[:,k,j])
                        x2p[res[l]['bt']] = np.nan
                        y2p[res[l]['bt']] = np.nan
                        ax[k,j].plot(x2p,y2p*1000./tot_exp_dur[k],c=colors[1+l])
                ax[k,j].legend()

            ax[k,0].set_ylabel('DN/pix/ks ({})'.format(filtershort_all[k]))
            ax[k,0].set_ylim(0.005,500)
            ax[k,0].set_yscale('log')

        ax[0,0].set_xlim(0,360)
        for j in range(nmodules):
            ax[0,j].set_title('Module {}'.format(self.SRnames[j]))
            ax[-1,j].set_xlabel('V3_PA')

        f.tight_layout()
        if save_figures_dir is not None:
            f.savefig(save_figures_dir+'PID{}_obsid{}_DN_sweep.pdf'.format(self.program,self.obsid))

    return tms, tcs


def compute_line(startx,starty,angle,length):

    anglerad = np.pi/180.*angle
    endx = startx + length*np.cos(anglerad)
    endy = starty + length*np.sin(anglerad)

    return np.array([startx,endx]), np.array([starty,endy])


class FilterInfo():
    '''
    Get info on filter wavelengths and bandpass
    '''    

    def __init__(self, filename= os.path.join(DATA_PATH, 'Filter_info.txt')):

        self.filter_table = pd.read_csv(filename, sep='\s+')

    def get_info(self, pupilshort, filtershort, key_info='Pivot'):
        if pupilshort == 'CLEAR':
            check_value = filtershort
        else:
            check_value = pupilshort

        BM = self.filter_table['Filter'] == check_value

        return self.filter_table.loc[BM,key_info].values[0]
