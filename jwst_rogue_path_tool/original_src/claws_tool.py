'''
Module to predict the presence of the stray-light artifacts know as claws
in NIRCam Imaging observations, specified in APT.

Authors
-------
    - Mario Gennaro
'''

from astroquery.simbad import Simbad
import pysiaf
from pysiaf.utils import rotations
from jwst_backgrounds import jbt

from apt_sql import Sqlfile

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


class apt_program():

    '''
    Class that handles the APT-program-level information.
    It can configure "observation" objects based on the desired observation ids,
    and can cal the observation.check_multiple_angles method to perform
    a check of stars in the susceptibility region for all the exposures of a
    given observation and for multiple observations of a given program
    '''

    def __init__(self,sqlfile,instrument='NIRCAM'):
        '''
        Parameters
        ----------
        sqlfile : str
            Path to an APT-exported sql file 

        catargs : dictionary
            parameters to be passed to the get_catalog method of an observation object
        '''       
                
        sql = Sqlfile(sqlfile)

        self.exptable = sql.table('exposures').to_pandas()
        try:
            self.targtable = sql.table('fixed_target').to_pandas()
        except:
            print('**** This is a moving target program, exiting ****')
            assert False
        self.nrctemptable = sql.table('nircam_templates').to_pandas()
        self.nestable = sql.table('nircam_exposure_specification').to_pandas()
        self.visittable = sql.table('visit').to_pandas()
        
        self.instrument = instrument
        self.PID = self.exptable.iloc[0]['program']
        self.siaf = pysiaf.Siaf(self.instrument)
        self.observations = []

    def configure_observation(self,obsid,catargs,smallregion=False,nes=None):
        '''
        Create an observation object

        Parameters
        ----------
        obsid: index of a Pandas dataframe
            Represents the observation ID of interest
            
        catargs: dictionary
            parameters to be passed to the get_catalog method of an observation object
            
        smallregion: boolean
            if True restricts the search to a smaller susceptibility region
            in the rogue path
            
        nes: integer
            specifies which exposure_spec_order_number to use. If None, the code will
            select the exposure_spec_order_number corresponding to the highest number 
            of expected counts (based on photon collecting time and zero point value)

        Returns
        ----------
        An observation object. Returns None if no exposures are present with the input
        obsid in the APT-exported sql file
        '''

        print('Adding obsid:',obsid)        

        if self.instrument == 'NIRCAM':
            prefix = 'NRC'

        obsskip = False
        nrcused = False
        nrcparallel = False
        BM = self.visittable['observation'] == obsid
        fvt = self.visittable[BM].iloc[0]
        if ((fvt['template'] != 'NIRCam Imaging') & (fvt['template'] != 'NIRCam Wide Field Slitless Spectroscopy')):
            if 'template_coord_parallel_1' in self.visittable.columns:
                if ((fvt['template_coord_parallel_1'] != 'NIRCam Imaging') & (fvt['template_coord_parallel_1'] != 'NIRCam Wide Field Slitless Spectroscopy')):
                    obsskip = True
                    if 'NIRC' in fvt['template']:
                        template_name = fvt['template']
                        nrcused = True
                    elif 'NIRC' in fvt['template_coord_parallel_1']:
                        template_name = fvt['template_coord_parallel_1']
                        nrcused = True
                else:
                    nrcparallel = True
                    nrcused = True

            else:
                obsskip = True
                if 'NIRC' in fvt['template']:
                    template_name = fvt['template']
                    nrcused = True
        else:
            nrcused = True
                
        if obsskip:
            if nrcused:
                print('**** The {} template is not supported, obsid {} will not be added ****'.format(template_name,obsid))
            else:
                print('**** obsid {} does not use NIRCam and will not be added ****'.format(obsid))
            return None
        else:
            if nrcparallel:
                print('**** NIRCam imaging used in parallel ****')
                
        

        BM = (self.exptable['observation'] == obsid) & (self.exptable['pointing_type'] == 'SCIENCE') \
              & (self.exptable['AperName'].str.contains(prefix))               
        
        if np.sum(BM)==0:
            print('No {} expsoures in this program for obsid {}'.format(self.instrument,obsid))
            return None
        print('Total number of {} exposures in this observation: {}'.format(self.instrument,np.sum(BM)))
        exptable_obs = self.exptable[BM]

        BM = (self.nestable['observation'] == obsid) 
        print('Total number of {} exposure specifications: {}'.format(self.instrument,np.sum(BM)))
        nestable_obs = self.nestable[BM]
        
        BM = (self.visittable['observation'] == obsid) 
        print('Total number of visits: {}'.format(np.sum(BM)))
        visittable_obs = self.visittable[BM]
        
        BM = self.nrctemptable['observation'] == obsid
        modules = self.nrctemptable[BM].iloc[0]['modules']


##### This part must go. Refactor the check one angle to check only for a single exposure specification

#       nes_ids = nestable['order_number']
#       pcds = np.empty(len(nes_ids))
#       zps = np.empty(len(nes_ids))
#       ps = np.empty(len(nes_ids),dtype=object)
#       fs = np.empty(len(nes_ids),dtype=object)
#       
#       zpc = zero_point_calc()
#       print('{:4} {:8} {:8} {:6} {:6} {:6}'.format('ID','PUPIL','FILTER','T_exp','   ZP','  K'))
#
#       for i,nes_id in enumerate(nes_ids):
#           BM = exptable_obs['exposure_spec_order_number'] == nes_id
#           pcds[i] = exptable_obs.loc[BM,'photon_collecting_duration'].values.astype(np.float_).sum()
#           
#           BM = nestable['order_number'] == nes_id
#           filtershort = nestable.loc[BM,'filter_short'].values[0]
#           if '_' in filtershort:
#               splt = filtershort.split('_')
#               pupilshort = splt[0]
#               filtershort = splt[1]
#           else:
#               pupilshort = 'CLEAR'
#           zps[i] = zpc.get_avg_zp(pupilshort,filtershort)
#           ps[i],fs[i] = pupilshort, filtershort
#           print('{:4} {:8} {:8} {:6.0f} {:6.2f} {:.2e}'.format(i,ps[i],fs[i],pcds[i],zps[i],pcds[i]*10**(zps[i]/2.5)))
#
#       if nes is None:                
#           cts_estimate = pcds*10**(zps/2.5)
#           ines = np.argmax(cts_estimate)
#           idxnes = nes_ids.index[ines]
#           nes = nes_ids[idxnes]
#           print('Exposure specification to use:{}, (idx={}); using PUPILSHORT={}, FILTERSHORT={}'.format(nes,ines,ps[ines],fs[ines]))
#       else:
#           print('Forcing use of exposure specification:{}, (idx={}); using PUPILSHORT={}, FILTERSHORT={}'.format(nes,nes+1,ps[nes],fs[nes]))
#       
#           
#
#       BM = exptable_obs['exposure_spec_order_number'] == nes
#       exptable_obs = exptable_obs[BM]
#
#
#       BM = nestable['order_number'] == nes
#       nestable_obs = nestable[BM]

        targetrowidx = self.targtable['target_id'] == exptable_obs.iloc[0]['target_id']
        target_ra  = self.targtable.loc[targetrowidx,'ra_computed'].values[0]
        target_dec = self.targtable.loc[targetrowidx,'dec_computed'].values[0]

        return observation(exptable_obs,nestable_obs,visittable_obs,target_ra,target_dec,modules,catargs,self.siaf,smallregion=smallregion)
        
    def add_observations(self,obsids=None,catargs={'inner_rad':8.,'outer_rad':12.,'sourcecat':'SIMBAD',
                                       'band':'K','maxmag':4.,'simbad_timeout':200,'verbose':True},smallregion=False):

        '''
        Configure multiple observations and append them to the self.observations list
        
        Parameters
        ----------
        obsid: list of Pandas indexes
            IDs of the observations to add. If None, all the opbservations in the prgram
            will be added

        catargs: dictionary
            parameters to be passed to the get_catalog method of an observation object
        '''

        if obsids is None:
            obsids = self.exptable['observation'].unique()
        for obsid in obsids:
            added = False
            for obs in self.observations:
                if obsid == obs.obsid:
                    print('Obsid {} already added'.format(obsid))
                    added = True
                    break
            if added == False:
                obshere = self.configure_observation(obsid,catargs,smallregion=smallregion)
                if obshere is not None:
                    self.observations.append(obshere)

    def check_observations(self,obsids=None,angstep=0.5,RP_padding=0):
    
        '''
        Convenience function to check multiple observations for stars in the 
        susceptibility region
        
        Parameters
        ----------
        obsid: list of Pandas indexes
            IDs of the observations to check. If None, all the opbservations in the prgram
            will be added

        angstep: float
            The resolution at which to scan the whole (0,360) range of PAs, in degrees
            
        RP_padding: float
            Extra padding around the susceptibility region (stars outside the nominal
            SR, but within RP_padding are flagged as "inside")
        '''
    
        if obsids is None:
            obsids = self.exptable['observation'].unique()
        for obsid in obsids:
            for obs in self.observations:
                if obs.obsid == obsid:
                    print('Checking obsid:',obsid)
                    obs.check_multiple_angles(angstep,RP_padding=0.)



class observation():

    '''
    Class that handles an indivdual observation.
    It contains info on each exposure pointing configuration, 
    retrieves a catalog within a certain annulus, and can check for stars 
    in the susceptibility region
    '''


    def __init__(self,exptable_obs,nestable_obs,visittable_obs,target_ra,target_dec,modules,catargs,siaf,smallregion=False):

        '''
        Parameters
        ----------
        exptable_obs: Pandas dataframe
            contains one row from the exposure table, per each exposure within the observation
            
        nestable_obs: Pandas dataframe
            contains one row from nircam_exposures_specification table for each exposure specification
             selected for this observation

        visittable_obs: Pandas dataframe
            contains one row from the visit table for each visit associated to this observation


        target_ra,target_dec: floats
            Coorindtaes of the target of the obsrvation in decimal degreess
            
        modules: list of strings
            name of the nircam modules configuration for the exposures within this observation
            (note that in NIRCam imaging, all the exposures of an observation must have the same configuration)
            
        catargs: dictionary
            parameters to be passed to the get_catalog method
            
        siaf: pysiaf.Siaf instance
            object containing the apertures info for NIRCam
        '''
        
        self.exptable_obs = exptable_obs
        self.nestable_obs = nestable_obs
        self.obsid = self.exptable_obs.iloc[0]['observation']
        self.program = self.exptable_obs.iloc[0]['program']
        self.target_ra = target_ra
        self.target_dec = target_dec
        self.modules = modules
        self.catargs = catargs
        self.smallregion = smallregion
        self.SRlist,self.SRnames = self.get_SRlist()
        self.efs = self.get_exposure_frames(siaf)
        self.catdf = self.get_catalog()

    def get_SRlist(self):
        '''
        Parameters
        ----------
        None
        
        Returns
        ----------
        SRlist: list of matplotlib.Path objects
            The list of susceptibility regions corresponding to the module used in this observation
        '''    

        if (self.modules == 'ALL') | (self.modules == 'BOTH'):
            SRlist = [sus_reg(module='A',small=self.smallregion),sus_reg(module='B',small=self.smallregion)]
            SRnames = ['A','B']
        else:
            if self.modules[0] == 'A':
                SRlist = [sus_reg(module='A',small=self.smallregion)]
                SRnames = ['A']
            elif self.modules[0] == 'B':
                SRlist = [sus_reg(module='B',small=self.smallregion)]
                SRnames = ['B']

        return SRlist,SRnames

    def get_exposure_frames(self,siaf):
        '''
        Parameters
        ----------
        siaf:
            a pysiaf.Siaf instance
        
        Returns
        ----------
        efs: list of exposure_frame objects
            The list of objects containing pointing info for each exposure within this observation
        '''    


        efs = []
        for i,row in self.exptable_obs.iterrows():
        
            BM = (self.nestable_obs['visit'] == row['visit']) & \
                 (self.nestable_obs['order_number'] == row['exposure_spec_order_number'])
            efs.append(exposure_frame(row,siaf,self.nestable_obs.loc[BM]))
        return efs

    def get_catalog(self):
        '''
        Parameters
        ----------
        None
        
        Returns
        ----------
        df: Pandas dataframe
            dataframe containing the coordinates of stars within an annulus centered on the
            observation target. The catalog characterisitics are based on the catargs dictionary
            
        '''    

        # Get the maximum relative offset between exposures and pad the catalog search radius
        ra_cen_sorted = np.sort(self.exptable_obs['ra_center_rotation'].values.astype(np.float_))
        dec_cen_sorted = np.sort(self.exptable_obs['dec_center_rotation'].values.astype(np.float_))

        max_ra_diff = np.abs(ra_cen_sorted[-1]-ra_cen_sorted[0])
        max_dec_diff = np.abs(dec_cen_sorted[-1]-dec_cen_sorted[0])

        max_delta = np.sqrt(np.sum(np.square([max_ra_diff,max_dec_diff])))
        
        inner_rad = self.catargs['inner_rad'] - max_delta
        outer_rad = self.catargs['outer_rad'] + max_delta
        
        if self.catargs['verbose']:
            print('Adopted inner and outer radius',inner_rad,outer_rad)

        # Retrieve a catalog
        if self.catargs['sourcecat'] == 'SIMBAD':
            df_in  = querysimbad(self.target_ra,self.target_dec,rad=inner_rad,
                                 band=self.catargs['band'],maxmag=self.catargs['maxmag'],simbad_timeout=self.catargs['simbad_timeout']).to_pandas()
            df_out = querysimbad(self.target_ra,self.target_dec,rad=outer_rad,
                                 band=self.catargs['band'],maxmag=self.catargs['maxmag'],simbad_timeout=self.catargs['simbad_timeout']).to_pandas()

            df = pd.concat([df_in, df_out]).drop_duplicates(keep=False)
            for i, row in df.iterrows():
                coord = SkyCoord(row['RA'], row['DEC'],unit=(u.hourangle, u.deg))
                df.loc[i,'RAdeg'] = coord.ra.deg
                df.loc[i,'DECdeg'] = coord.dec.deg
                
        if self.catargs['sourcecat'] == '2MASS':
            df = pd.read_csv(self.catargs['2MASS_filename'])
            BM = df[self.catargs['band']] < self.catargs['maxmag']
            df = df[BM]
            
            c1 = SkyCoord(self.target_ra*u.deg, self.target_dec*u.deg, frame='icrs')
            c2 = SkyCoord(df['ra']*u.deg, df['dec']*u.deg, frame='icrs')
            sep = c1.separation(c2)
            BM = (sep.deg < outer_rad) & (sep.deg > inner_rad)
            
            df=df[BM]
            df.rename(columns={'ra': 'RAdeg', 'dec': 'DECdeg'},inplace=True)
            
        
        return df
        
     
    def check_multiple_angles(self,angstep,filtershort=None,RP_padding=0.):

        '''
        Convenience method to check multiple angles at once, for all
        exposures within this observation
        
        Parameters
        ----------
        angstep: float
            The resolution at which to scan the whole (0,360) range of PAs, in degrees

        filtershort: string
            name of the filter in use, will be used to downselect the exposures.
            if None will be defaulted to the first filter in the nircam_expsoures_specification
            table

        RP_padding: float
            Extra padding around the susceptibility region (stars outside the nominal
            SR, but within RP_padding are flagged as "inside")
        '''

        if filtershort is None:
            filtershort = self.nestable_obs['filter_short'].values[0]

        efs_here = [ef for ef in self.efs if ef.nestable_row['filter_short'].values[0] == filtershort]

        self.angstep = angstep
        self.RP_padding = RP_padding
        self.attitudes = np.arange(0.,360.,angstep)
        self.IN = np.empty([len(self.catdf),self.attitudes.size,len(self.SRlist),len(efs_here)],dtype=np.bool_)
        self.V2 = np.empty([len(self.catdf),self.attitudes.size,len(efs_here)])
        self.V3 = np.empty([len(self.catdf),self.attitudes.size,len(efs_here)])
        self.good_angles = np.zeros([self.attitudes.size,len(efs_here)],dtype=np.bool_)

        for i,att in enumerate(self.attitudes):
            IN_one, V2_one, V3_one, check_one = self.check_one_angle(att,filtershort,RP_padding=self.RP_padding)
            self.V2[:,i,:],self.V3[:,i,:] = V2_one,V3_one
            self.IN[:,i,:,:] = IN_one
            self.good_angles[i,:] = check_one
            
        V3PA_validranges_starts= []
        V3PA_validranges_ends  = []

        for k in range(len(efs_here)):
            change = np.where(self.good_angles[:-1,k] != self.good_angles[1:,k])[0]
            
            if change.size >0:
            
                if self.good_angles[change[0],k]:
                    change = np.roll(change,1)
        
                V3PA_validranges_starts.append(self.angstep*change[::2])
                V3PA_validranges_ends.append(self.angstep*change[1::2])
        else:
             V3PA_validranges_starts.append(None)
             V3PA_validranges_ends.append(None)
        
        
        
        self.good_angles_obs = np.all(self.good_angles,axis=1)

        V3PA_validranges_obs_starts= []
        V3PA_validranges_obs_ends  = []
        
        change = np.where(self.good_angles_obs[:-1] != self.good_angles_obs[1:])[0]
        if change.size >0:
            if self.good_angles_obs[change[0]]:
                change = np.roll(change,1)
        
            V3PA_validranges_obs_starts = self.angstep*change[::2]
            V3PA_validranges_obs_ends = self.angstep*change[1::2]
        else:
            V3PA_validranges_obs_starts = None
            V3PA_validranges_obs_ends = None
        
        
        self.V3PA_validranges_starts = V3PA_validranges_starts
        self.V3PA_validranges_ends = V3PA_validranges_ends
        self.V3PA_validranges_obs_starts = V3PA_validranges_obs_starts
        self.V3PA_validranges_obs_ends = V3PA_validranges_obs_ends


    def check_one_angle(self,att,filtershort,RP_padding=0.):

        '''
        Method to check for the presence of stars in the susceptibility
        region at a fixed angle, for all exposures of a given observation
        
        Parameters
        ----------
        att: float
            position angle to check, in degrees
            
        filtershort: string
            name of the filter in use, will be used to downselect the exposures
            
        RP_padding: float
            Extra padding around the susceptibility region (stars outside the nominal
            SR, but within RP_padding are flagged as "inside")
            
        Returns
        ---------
        IN_one: numpy boolean array (catalog size x number of SR region x number of exposures in the observation)
            True values indicate stars that are in (one of) the SRs for one of the exposures
        V2_one, V3one: numpy array (catalog size x number of exposures in the observation)
            V2, V3 coordinates (jn deg) for a given stars and a given exposure
        check_one: numpy boolean array (number of exposures in the observation)
            True if any star is in either SRs for a given exposure
        '''

        efs_here = [ef for ef in self.efs if ef.nestable_row['filter_short'].values[0] == filtershort]

        IN_one = np.empty([len(self.catdf),len(self.SRlist),len(efs_here)],dtype=np.bool_)
        V2_one = np.empty([len(self.catdf),len(efs_here)])
        V3_one = np.empty([len(self.catdf),len(efs_here)])
        check_one = np.zeros(len(efs_here),dtype=np.bool_)

        for k,ef in enumerate(efs_here):
            ef.define_attitude(att)
            V2_one[:,k],V3_one[:,k] = ef.V2V3_at_one_attitude(self.catdf['RAdeg'],self.catdf['DECdeg'])
 
            for j,SR in enumerate(self.SRlist):
                IN_one[:,j,k] = SR.V2V3path.contains_points(np.array([V2_one[:,k],V3_one[:,k]]).T,radius=RP_padding)
            if len(self.SRlist) > 1:
                if ~(np.any(IN_one[:,0,k]) | np.any(IN_one[:,1,k])):
                    check_one[k] =  True
            else:
                if ~(np.any(IN_one[:,0,k])):
                    check_one[k] =  True
 
        return IN_one, V2_one, V3_one, check_one

        
    def plot_obs_field(self,bandpass):
        '''
        Method to plot the stars in the catalog
        
        Parameters
        ----------
        bandpass: string
            bandpass to use for color-coding the stars
        '''

        f,ax = plt.subplots(1,1)
        
        ax.scatter(self.catdf['RAdeg'],self.catdf['DECdeg'],c=self.catdf[bandpass])
        for ef in self.efs:
            ax.scatter(ef.raRef,ef.decRef,marker='o',c='orange')
        ax.scatter(self.target_ra,self.target_dec,marker='X',c='red')
        
        
        ax.axis('equal')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.invert_xaxis()
        f.tight_layout()
        
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

    def fixed_angle(self,att,RP_padding=0.,smooth=None,draw_allexp=True,draw_summary=True,nrows=2,ncols=3,savefilenames=[None,None],metainfo=None):
    
        '''
        Method to check a single angle and return diagnistic plot and info.
        Useful for checking executed observations.
        
        Parameters
        ----------
        att: float
            position angle to check, in degrees
            
        RP_padding: float
            Extra padding around the susceptibility region (stars outside the nominal
            SR, but within RP_padding are flagged as "inside")
            
        draw_allexp: boolean
            flag to enable/disable plotting of the per-exposure results

        draw_summary: boolean
            flag to enable/disable plotting of the overall results
            
        nrows, ncols: integers
            number of rows and columns in the grid plot
            
        savefilename: [file path,file path]
            saves the figures to these location
            (first element for per-expsoure, second element for global plots).
            If None it doesn't save one file 
            
        Returns
        ---------
        report_dict:
            dictionary containing some metinfo and a pands dataframe with the True/False 
            conditions for each stars for each exposure for each susceptibility region
        
        '''

        filtershort_all = []
        tot_exp_dur = []

        exp_spec_orders = self.nestable_obs['order_number'].unique()
        for exp_spec_order in exp_spec_orders:
            BM = self.exptable_obs['exposure_spec_order_number'] == exp_spec_order
            tot_exp_dur.append(self.exptable_obs.loc[BM,'photon_collecting_duration'].values.astype(np.float_).sum())
            BM = self.nestable_obs['order_number'] == exp_spec_order
            filtershort_all.append(self.nestable_obs.loc[BM,'filter_short'].values[0])
        tot_exp_dur = np.asarray(tot_exp_dur,dtype=np.float_)

        efs_here = [ef for ef in self.efs if ef.nestable_row['filter_short'].values[0] == filtershort_all[0]]

        # It is sufficient to check all the exposures for a single filter, because all the various exposure specifications
        # will results in an identical number of exposures, with identical pointings across the whole observation
        
        IN_one, V2_one, V3_one, check_one = self.check_one_angle(att,filtershort_all[0],RP_padding=RP_padding)
        colors = ['tomato','olivedrab']
        
        rpi_A = rogue_path_intensity(module='A',smooth=smooth)
        rpi_B = rogue_path_intensity(module='B',smooth=smooth)
        
        intensities_A = rpi_A.get_intensity(V2_one,V3_one)
        intensities_B = rpi_B.get_intensity(V2_one,V3_one)
        
        avg_intensity_A = np.mean(intensities_A,axis=1)
        avg_intensity_B = np.mean(intensities_B,axis=1)
        
        avg_V2 = np.mean(V2_one,axis=1)
        avg_V3 = np.mean(V3_one,axis=1)
        zp = 17 

        emp_zp = emp_zero_point()
        zp_vega_all_A = []
        zp_vega_all_B = []
        ground_bands = []

        filternames = []
        pupilnames = []
        for filtershort in filtershort_all:
            if '+' in filtershort:
                splt = filtershort.split('+')
                pupilshort = splt[0]
                filtershort = splt[1]
            elif '_' in filtershort:
                splt = filtershort.split('_')
                pupilshort = splt[0]
                filtershort = splt[1]
            else:
                pupilshort = 'CLEAR'
            zp_vega_all_A.append(emp_zp.get_emp_zp('A',pupilshort,filtershort))
            zp_vega_all_B.append(emp_zp.get_emp_zp('B',pupilshort,filtershort))
            ground_bands.append(emp_zp.get_ground_band(pupilshort,filtershort,catalog=self.catargs['sourcecat']))
            filternames.append(filtershort)
            pupilnames.append(pupilshort)

        zp_vega_all_A = np.asarray(zp_vega_all_A,dtype=np.float_)
        zp_vega_all_B = np.asarray(zp_vega_all_B,dtype=np.float_)

        fict_mag_A = {}
        fict_mag_B = {}

        if self.catargs['sourcecat'] == '2MASS':
            bands = ['j_m','h_m','k_m']
        else:
            bands = ['J','H','K']

        for band in bands:
            fict_mag_A[band] = self.catdf[band] - 2.5*np.log10(avg_intensity_A) + zp
            fict_mag_B[band] = self.catdf[band] - 2.5*np.log10(avg_intensity_B) + zp
            BM_A = fict_mag_A[band] == np.inf
            fict_mag_A[band][BM_A] = 99.
            BM_B = fict_mag_B[band] == np.inf
            fict_mag_B[band][BM_B] = 99.

        newdf_columns = ['RAdeg','DECdeg']+bands
        newdf = self.catdf[newdf_columns].copy()
        newdf['avg_intensity_A'] = avg_intensity_A
        newdf['avg_intensity_B'] = avg_intensity_B
        newdf['avg_V2'] = avg_V2
        newdf['avg_V3'] = avg_V3        
        for band in bands:
            newdf[band+'_fict_mag_A'] = fict_mag_A[band]
            newdf[band+'_fict_mag_B'] = fict_mag_B[band]


        totmag_A = {}
        for band in bands:
            BM = ~pd.isnull(newdf[band+'_fict_mag_A'])
            totmag_A[band] = -2.5*np.log10(np.sum(np.power(10,-0.4*newdf.loc[BM,band+'_fict_mag_A'].values)))
        totcts_A = []
        for i, (zp_vega, gb) in enumerate(zip(zp_vega_all_A,ground_bands)):
            totcts_A.append(10**((zp_vega-totmag_A[gb])/2.5) * tot_exp_dur[i])

        totmag_B = {}
        for band in bands:
            BM = ~pd.isnull(newdf[band+'_fict_mag_B'])
            totmag_B[band] = -2.5*np.log10(np.sum(np.power(10,-0.4*newdf.loc[BM,band+'_fict_mag_B'].values)))
        totcts_B = []
        for i, (zp_vega, gb) in enumerate(zip(zp_vega_all_B,ground_bands)):
            totcts_B.append(10**((zp_vega-totmag_B[gb])/2.5) * tot_exp_dur[i])

        report_dict = {'V3PA':att,
                       'catalog':newdf,
                       'RP_padding':RP_padding,
                       'totmag_A':totmag_A,
                       'totmag_B':totmag_B,
                       'totcts_A':totcts_A,
                       'totcts_B':totcts_B,
                       'tot_exp_dur':tot_exp_dur,
                       'zp_vega_all_A':zp_vega_all_A,
                       'zp_vega_all_B':zp_vega_all_B,
                       'filtershort_all':filtershort_all,
                       'filternames':filternames,
                       'pupilnames':pupilnames,
                       'bands':bands
                       }

        if draw_allexp:
            f,axs = plt.subplots(nrows,ncols,figsize=(4*ncols,3.5*nrows))
            for k,(ef,ax) in enumerate(zip(efs_here,axs.reshape(-1))):
                for i,(SR,SRn) in enumerate(zip(self.SRlist,self.SRnames)):
                    onepatch = patches.PathPatch(SR.V2V3path,  lw=2,alpha=0.5, color=colors[i], label=SRn)
                    ax.add_patch(onepatch)

                if len(self.SRlist) > 1:
                    INcol = IN_one[:,0,k]|IN_one[:,1,k]
                else:
                    INcol = IN_one[:,0,k]

                sz = (35./(2.5+self.catdf[self.catargs['band']]))**2.
                BM = sz >300
                sz[BM] =300
                BM = sz <5
                sz[BM] =5

                ax.scatter(V2_one[:,k],V3_one[:,k],c=INcol,s=sz)
                ax.set_xlim(3.25,-3.25)
                ax.set_ylim(7.5,12.5)
                ax.set_xlabel('V2')
                ax.set_ylabel('V3')
                ax.legend()
                ax.set_title('Expnum: {}'.format(k+1))

            f.suptitle('Obsid: {}'.format(self.obsid))
            f.tight_layout()
        
            if savefilenames[0] is not None:
                f.savefig(savefilenames[0])
        

###ADD PID and exptime to plot

        if draw_summary:
            f,ax = plt.subplots(1,1,figsize=(8.5,4))

            count_message = 'Mod     Filt           t$_{{ph.c.}}$   DN/px/ks \n'
            count_message_A = None
            count_message_B = None

            for i,(SR,SRn) in enumerate(zip(self.SRlist,self.SRnames)):
                if SRn == 'A':
                    label = 'Mod A:'
                    for band in bands:
                        label += '\n{}$_{{tot}}$:{:6.2f}'.format(band,totmag_A[band])
                    count_message_A = ''
                    for flt,t,tc in zip(filtershort_all,tot_exp_dur,totcts_A):
                        count_message_A += '  {:2}   {:11} {:5.1f}  {:6.3f}\n'.format('A',flt,t,1000.*tc/t)

                else:
                    label = 'Mod B:'
                    for band in bands:
                        label += '\n{}$_{{tot}}$:{:6.2f}'.format(band,totmag_B[band])

                    count_message_B = ''
                    for flt,t,tc in zip(filtershort_all,tot_exp_dur,totcts_B):
                        count_message_B += '  {:2}   {:11} {:5.1f}  {:6.3f}\n'.format('B',flt,t,1000.*tc/t)

                onepatch = patches.PathPatch(SR.V2V3path,  lw=2,alpha=0.5, color=colors[i], label=label)
                ax.add_patch(onepatch)

            if count_message_A is not None:
                count_message+= count_message_A
            if count_message_B is not None:
                count_message+= count_message_B
            

            if len(self.SRlist) > 1:
                IN_0 = IN_one[:,0,:]
                IN_1 = IN_one[:,1,:]
                
                IN_both   = np.all(IN_0,axis=1) & np.all(IN_1,axis=1)
                IN_0_only = np.all(IN_0,axis=1) & ~(np.all(IN_1,axis=1))
                IN_1_only = np.all(IN_1,axis=1) & ~(np.all(IN_0,axis=1))
                
                INcol = np.all(IN_0,axis=1)|np.all(IN_1,axis=1)
                
                fict_mag_draw = np.empty_like(newdf[self.catargs['band']+'_fict_mag_A'])
                fict_mag_draw[IN_0_only] = newdf.loc[IN_0_only,self.catargs['band']+'_fict_mag_A']
                fict_mag_draw[IN_1_only] = newdf.loc[IN_1_only,self.catargs['band']+'_fict_mag_B']
                fict_mag_draw[IN_both] = 0.5*(newdf.loc[IN_both,self.catargs['band']+'_fict_mag_A']+ newdf.loc[IN_both,self.catargs['band']+'_fict_mag_B'])
                
            else:
                IN_0 = IN_one[:,0,:]
                INcol = np.all(IN_0,axis=1)
                fict_mag_draw = newdf[self.catargs['band']+'_fict_mag_B']

            sz = (35./(-15+fict_mag_draw))**2.
            BM = sz >300
            sz[BM] =300
            BM = sz< 5
            sz[BM] = 5 

            ax.scatter(avg_V2,avg_V3,c=INcol,s=sz)
            ax.set_xlim(3.25,-3.25)
            ax.set_ylim(6.75,12.75)
            ax.set_xlabel('V2')
            ax.set_ylabel('V3')
            ax.legend(loc=2,facecolor='lightgray',edgecolor='darkgray',framealpha=0.5,fontsize=10,ncol=2)
 #           ax.set_title('Averaged over the {}-th exp. spec.'.format(self.nestable_obs['order_number'].values[0]),fontsize=12)

            stbox = dict(boxstyle="round",fc='lightgray',ec='darkgray',alpha=0.5)
 #           ax.text(3.1, 11.33,'{}+{}\n ZP$_{{Vega}}$={:5.2f}\n t$_{{ph.c.}} =$ {:.0f}s'.format(pupilshort,filtershort,zp_vega,tot_exp_dur), ha='left', fontsize=10,bbox=stbox)
 
            lcm = len(count_message.splitlines())
            fts = 10
            cmxloc,cmyloc = -3.5, 12.725
            
            if lcm > 10:
                fts = 9
                if lcm > 16:
                    fts = 8
                    if lcm > 20:
                        splt = count_message.splitlines()
                        hdr = splt[0]
                        hl = (lcm-1)//2
                        count_message = hdr[:-1]+'   '+hdr[:-1]+'\n'
                        for l in range(hl):
                            count_message += splt[1+l][:-1]+'      '+splt[1+l+hl][:-1]+'\n'
            
            ax.text(cmxloc, cmyloc,count_message, ha='left', va='top', fontsize=fts,bbox=stbox)

            if metainfo is not None:
                annotation = 'Detectors: {}, Claws: {}'.format(metainfo['detector'].strip(),metainfo['claws'].strip())
                for s,string in enumerate(metainfo['notes'].split(';')):
                    annotation+='\n'
                    if s==0:
                        annotation+='Notes:'
                    annotation+=string.strip()
                ax.text(3.1, 7.0, annotation, ha='left', fontsize=10,bbox=stbox)
            
            f.suptitle('PID: {}, Obsid: {}, PA:{:6.2f}'.format(self.program,self.obsid,att))
            f.tight_layout()
        
            if savefilenames[1] is not None:
                f.savefig(savefilenames[1])

        
        return report_dict
        
        
    def plot_observations_checks(self,nrows=2,ncols=3,verbose=True,filtershort=None):
    
        '''
        Method to plot some summary results after running self.check_observations.
        It plots the claws-unaffected angles for each exposure and a summary of
        claws-unaffected angles over the whole observation
        
        Parameters
        ----------
        nrows, ncols: integers
            number of rows and columns in the grid plot
        '''

        if filtershort is None:
            filtershort = self.nestable_obs['filter_short'].values[0]

        efs_here = [ef for ef in self.efs if ef.nestable_row['filter_short'].values[0] == filtershort]

        #### The exposure-level plots
        f1,axs = plt.subplots(nrows,ncols,figsize=(4*ncols,3.5*nrows))
        for k,(ef,ax) in enumerate(zip(efs_here,axs.reshape(-1))):
            ax.scatter(self.catdf['RAdeg'],self.catdf['DECdeg'],c='deeppink')
            ax.scatter(self.target_ra,self.target_dec,marker='X',c='red')
            ax.scatter(ef.raRef, ef.decRef,marker='X',c='orange')
            ax.axis('equal')
            ax.set_xlabel('RA')
            ax.set_ylabel('Dec')
            ax.invert_xaxis()
            ax.set_title('Expnum: {}'.format(k+1))

            for i,att in enumerate(self.attitudes):
                if self.good_angles[i,k] == True:
                    ef.define_attitude(att)
                    for SR in self.SRlist:
                        SR_RA,SR_DEC = rotations.tel_to_sky(ef.attitude, 3600*SR.V2V3path.vertices.T[0],3600*SR.V2V3path.vertices.T[1])
                        SR_RAdeg,SR_DECdeg = SR_RA.value*180./np.pi,SR_DEC.value*180./np.pi
                        RADEC_path = Path(np.array([SR_RAdeg,SR_DECdeg]).T,SR.V2V3path.codes)
                        RADEC_patch = patches.PathPatch(RADEC_path,  lw=2,alpha=0.05)
                        ax.add_patch(RADEC_patch)

            draw_angstep = self.angstep
            for s,e in zip(self.V3PA_validranges_starts[k],self.V3PA_validranges_ends[k]):
                wd = patches.Wedge((ef.raRef, ef.decRef), 5.5, 90-e-0.5*draw_angstep, 90-s+0.5*draw_angstep,width=.5)
                wd.set(color='darkseagreen')

                ls = compute_line(ef.raRef, ef.decRef,90-s+0.5*draw_angstep,5.75)
                le = compute_line(ef.raRef, ef.decRef,90-e-0.5*draw_angstep,5.75)
                lm = compute_line(ef.raRef, ef.decRef,90-0.5*(s+e),7.)

                ax.add_artist(wd)
                ax.plot(ls[0],ls[1],color='darkseagreen')
                ax.plot(le[0],le[1],color='darkseagreen')
                ax.text(lm[0][1],lm[1][1], '{}-{}'.format(s,e), fontsize=10,horizontalalignment='center',verticalalignment='center')

            ax.set_title('Expnum: {}'.format(k+1))
        f1.suptitle('Obsid: {}'.format(self.obsid))
        f1.tight_layout()

        #### The observation-level plots
        f2,ax2 = plt.subplots(1,1,figsize=(6,6))
        ax2.scatter(self.catdf['RAdeg'],self.catdf['DECdeg'],c='deeppink')
        ax2.scatter(self.target_ra,self.target_dec,marker='X',c='red')
        ax2.axis('equal')
        ax2.set_xlabel('RA')
        ax2.set_ylabel('Dec')
        ax2.invert_xaxis()

        draw_angstep = self.angstep
        if verbose == True:
            print('*** Valid ranges ****')
        
        for s,e in zip(self.V3PA_validranges_obs_starts,self.V3PA_validranges_obs_ends):
            wd = patches.Wedge((self.target_ra,self.target_dec), 5.5, 90-e-0.5*draw_angstep, 90-s+0.5*draw_angstep,width=.5)
            wd.set(color='darkseagreen')

            ls = compute_line(self.target_ra,self.target_dec,90-s+0.5*draw_angstep,5.75)
            le = compute_line(self.target_ra,self.target_dec,90-e-0.5*draw_angstep,5.75)
            lm = compute_line(self.target_ra,self.target_dec,90-0.5*(s+e),7.)

            ax2.add_artist(wd)
            ax2.plot(ls[0],ls[1],color='darkseagreen')
            ax2.plot(le[0],le[1],color='darkseagreen')
            ax2.text(lm[0][1],lm[1][1], '{}-{}'.format(s,e), fontsize=10,horizontalalignment='center',verticalalignment='center')
            
            if verbose == True:
                print('PA Start -- PA End: {} -- {}'.format(s,e))


        ax2.set_title('Summary for obsid {}'.format(self.obsid))
        f2.tight_layout()
        
        return f1,f2

#### Write a method to predict the intensity of a claw based on Scott R. intensity map and
#### on the star's magnitude and exp time

class exposure_frame():
    '''
    The main class to handle pointing info and rotation for an individual exposure
    '''

    def __init__(self,exptable_row,target_ra,target_dec,siaf,nestable_row):
    
        '''
        Parameters
        ----------
        exptable_row: single row in a Pandas dataframe
            contains the pointing info on this specific exposures
            
        siaf: instance of pysiaf.Siaf
            used to extract the info on the aperture used in this exposure
            
        nestable_row: single row in a Pandas dataframe
            contains info on the expsoure specification from which the exposure
             in question is generated
        '''

        self.exptable_row = exptable_row
        self.nestable_row = nestable_row
        self.V2target = np.float_(self.exptable_row['v2'])
        self.V3target = np.float_(self.exptable_row['v3'])
        self.ratarget = np.float_(target_ra)
        self.dectarget = np.float_(target_dec)
        

    def define_attitude(self,v3pa,usetarget=True):
        '''
        Define an attitude matrix (pysiaf.rotations)

        Parameters
        ----------
        v3pa:
            position angle of the v3 axis
        '''
        self.attitude = rotations.attitude(self.V2target,self.V3target,self.ratarget,self.dectarget,v3pa)
    
    def V2V3_at_one_attitude(self,radeg,decdeg,verbose=False):
        '''
        Compute V2,V3 locations of stars at a given attitude

        Parameters
        ----------
        radeg, decdeg: lists of floats
            stellar coordinates in decimal degrees
           
        Returns
        ---------
        v2,v3 positions in degrees

       '''

        v2rads,v3rads = rotations.sky_to_tel(self.attitude, radeg, decdeg, verbose=verbose)
        return v2rads.value*180./np.pi,v3rads.value*180./np.pi


'''
Function to put together a "query by criteria" SIMBAD query 
and return an astropy Table with the results.
Query criteria here are a circle radius and a faint magnitude limit
based on a user-selectable bandpass
'''

def querysimbad(ra,dec,rad=1,band='K',maxmag=6.,simbad_timeout=200):

    Simbad.TIMEOUT = simbad_timeout
    Simbad.reset_votable_fields()
    
    for filtername in ['J','H','K']:
        for prop in ['','_bibcode','_error','_name','_qual','_system','_unit','data']:
            field = 'flux{}({})'.format(prop,filtername)
            Simbad.add_votable_fields(field)
    
    if ra >=0.:
        ras = '+'
    else:
        ras = '-'
    if dec >=0.:
        decs = '+'
    else:
        decs = '-'
    
    crit = 'region(circle, ICRS, {}{} {}{},{}d) & ({}mag < {})'.format(ras,ra,decs,dec,rad,band,maxmag)
    print(crit)
    t = Simbad.query_criteria(crit)
    return t


'''
Function to obtain the (v1_ra,v1_dec,v3_pa) of a visit.
If a header is passed, it will get the info from it, 
if the visit_id is passed, it will query the PPSDB visit_execution table to obtain the same info
'''
def get_pointing_info(header=None, visit_id=None):
    
    if (header is not None) and (visit_id is not None):
        print('Cannot specify both a header and a visit_id')
        assert False

    if header is not None:
        return header['RA_V1'],header['DEC_V1'],header['PA_V3']

    if visit_id is not None:
        pass
        # Need to write ppsdb query but need to set up the env correctly firs


    
## Add an option for the small SR


'''Convenience class that creates a matplotlib.Path with the Rogue Path
susceptibility zone vertices for a given NIRCam module'''
class sus_reg():

    def __init__(self,module='A',small=False):
        self.small = small
        self.module = module
        self.V2V3path = self.get_path()


    def get_path(self):
        if self.module== 'A':
            if self.small == False:
                V2list = [2.64057,  2.31386,  0.47891,  0.22949, -0.04765, -0.97993, -0.54959,  0.39577,  0.39577,  1.08903,  1.56903,  2.62672,  2.64057]
                V3list = [10.33689,  10.62035,  10.64102,  10.36454,  10.65485,  10.63687,   9.89380,   9.47981,   9.96365,   9.71216,   9.31586,   9.93600,  10.33689]
            else: 
                V2list = [2.28483,  0.69605,  0.43254,  0.57463,  0.89239,  1.02414,  1.70874,  2.28483,  2.28483]
                V3list = [10.48440,  10.48183,  10.25245,  10.12101,  10.07204,   9.95349,  10.03854,  10.04369,  10.48440]


        else:
            if self.small == False:
                V2list = [0.52048,   0.03549,  -0.28321,  -0.49107,  -2.80515,  -2.83287,  -1.58575,  -0.51878,  -0.51878,  -0.40792,   0.11863,   0.70062,   0.52048]
                V3list = [ 10.32307, 10.32307, 10.01894, 10.33689, 10.33689,  9.67334,  9.07891,  9.63187,  8.99597,  8.96832,  9.21715,  9.70099, 10.32307]
            else:
                V2list = [-0.96179, -1.10382, -2.41445, -2.54651, -2.54153, -2.28987, -1.69435, -1.46262, -1.11130, -0.95681, -0.59551, -0.58306, -0.96179]
                V3list = [10.03871, 10.15554, 10.15554, 10.04368,  9.90945,  9.82741,  9.76030,  9.64347,  9.62855,  9.77273,  9.88459, 10.07848, 10.03871]

        V2list = [-1.*v for v in V2list]

        verts = []
        for xx,yy in zip(V2list,V3list):
            verts.append((xx,yy))
        codes = [Path.MOVETO]
        for _ in verts[1:-1]:
            codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)
        return Path(verts, codes)


class rogue_path_intensity():
    '''
    Use the FITS files provided by Scott Rohrbach to get the intensity of the
    susceptibility zone at a given V2,V3
    '''
    
    def __init__(self,module='A',smooth=None):
    
        self.module=module
        if self.module == 'A':
            filename = 'Rogue path NCA.fits'
        else:
            filename = 'Rogue path NCB.fits'
        self.filename = 'path/to/future/datadir'+filename
        self.fh = fits.getheader(self.filename)
        
        if smooth is not None:
            fd = fits.getdata(self.filename)
            self.fd = gaussian_filter(fd, sigma=smooth)
        else:
            self.fd = fits.getdata(self.filename)
            
        self.fd[:,:60] = 0.
        self.fd[:,245:] = 0.
        self.fd[:85,:] = 0.
        self.fd[160:,:] = 0.
        
        
                
    def get_intensity(self,V2,V3):
    
#        x = (V2-self.fh['AAXISMAX'])/(self.fh['AAXISMIN']-self.fh['AAXISMAX'])*self.fh['NAXIS1']
        x = (V2-self.fh['AAXISMIN'])/(self.fh['AAXISMAX']-self.fh['AAXISMIN'])*self.fh['NAXIS1']
        y = (V3-self.fh['BAXISMIN'])/(self.fh['BAXISMAX']-self.fh['BAXISMIN'])*self.fh['NAXIS2']
        
        xint = np.floor(x).astype(np.int_)
        yint = np.floor(y).astype(np.int_)
        
        BM1 = xint<0
        BM2 = yint<0
        BM3 = xint>=self.fh['NAXIS1']
        BM4 = yint>=self.fh['NAXIS2']
        BM = BM1|BM2|BM3|BM4

        xint[BM] = 0
        yint[BM] = 0
        
        return(self.fd[yint,xint])
        

def compute_line(startx,starty,angle,length):

    anglerad = np.pi/180.*angle
    endx = startx + length*np.cos(anglerad)
    endy = starty + length*np.sin(anglerad)

    return np.array([startx,endx]), np.array([starty,endy])


class zero_point_calc():

    '''
    Get the average quantities (eg zp_vega og PHOTMJSR) over SCAs for a PUPIL+FILTER combo
    '''
    
    def __init__(self,filename='/path/to/future/datadir/' + 'NRC_ZPs_0995pmap.txt'):

        df = pd.read_csv(filename,skiprows=4,sep='|',names= [y.strip() for y in 'dum | pupil+filter |      sca | PHOTMJSR | zp_vega |  vega_Jy | zp_AB | mean_pix_sr | dum2'.split('|')])
        df.drop(columns=['dum','dum2'],inplace=True)
        df['pupil+filter'] = df['pupil+filter'].map(str.strip)
        self.zp_table=df

    def get_avg_quantity(self,pupilshort,filtershort,quantity='PHOTMJSR'):

        BM = self.zp_table['pupil+filter'] == '{}+{}'.format(pupilshort,filtershort)
        return np.mean(self.zp_table.loc[BM,quantity])

class emp_zero_point():

    '''
    Get an empirical zp given the module and the PUPIL+FILTER combo
    '''
    
    def __init__(self):
    
        self.zp_A = {'CLEAR+F070W': 7.,
                     'CLEAR+F090W': 8.,
                     'CLEAR+F182M': 9.7436,
                     'CLEAR+F187N': 7.2,
                     'CLEAR+F140M':9.,
                     'F162M+F150W2':7.5,
                     'CLEAR+F212N': 8.1523,
                     'CLEAR+F210M': 9.5981,
                     'F164N+F150W2': 5.5,
                     'CLEAR+F115W': 9.3934,
                     'CLEAR+F150W': 10.2889,
                     'CLEAR+F150W2': 10.88-0.37-0.1-0.0058,
                     'CLEAR+F200W': 10.8321} 

        self.zp_B = {'CLEAR+F070W': 8,
                     'CLEAR+F090W': 10.8491,
                     'CLEAR+F182M': 10.9,
                     'CLEAR+F187N': 8.3,
                     'CLEAR+F140M':9.5,
                     'F162M+F150W2':8.5,
                     'CLEAR+F212N': 8.2,
                     'CLEAR+F210M': 10.7,
                     'F164N+F150W2': 11.0419,
                     'CLEAR+F115W': 9.8801,
                     'CLEAR+F150W': 11.8580,
                     'CLEAR+F150W2': 12.5890,
                     'CLEAR+F200W': 12.0125}  

        self.match2MASS = {'CLEAR+F070W': 'j_m',
                           'CLEAR+F090W': 'j_m',
                           'CLEAR+F182M': 'h_m',
                           'CLEAR+F187N': 'h_m',
                           'CLEAR+F115W': 'h_m',
                           'CLEAR+F140M': 'h_m',
                           'F162M+F150W2': 'h_m',
                           'CLEAR+F150W': 'h_m',
                           'CLEAR+F150W2': 'h_m',
                           'F164N+F150W2': 'h_m',
                           'CLEAR+F200W': 'k_m',
                           'CLEAR+F212N': 'k_m',
                           'CLEAR+F210M': 'k_m'}

        self.matchSIMBAD = {'CLEAR+F090W': 'I',
                            'CLEAR+F090W': 'J',
                            'CLEAR+F182M': 'H',
                            'CLEAR+F187N': 'H',
                            'CLEAR+F140M': 'H',
                            'F162M+F150W2': 'H',
                            'CLEAR+F115W': 'H',
                            'CLEAR+F150W': 'H',
                            'CLEAR+F150W2': 'H',
                            'F164N+F150W2': 'H',
                            'CLEAR+F200W': 'K',
                            'CLEAR+F212N': 'K',
                            'CLEAR+F210M': 'K'}


    def get_emp_zp(self,module,pupilshort,filtershort):

        if module == 'A':
            return self.zp_A['{}+{}'.format(pupilshort,filtershort)]
        elif module == 'B':
            return self.zp_B['{}+{}'.format(pupilshort,filtershort)]
        else:
            print('Wrong module value', module)
            assert False

    '''
    This method returns the 2MASS/Simbad band to be associated with each SW NIRCam band
    to find the appropriate magnitude values for estimating counts
    '''
    
    def get_ground_band(self,pupilshort,filtershort,catalog='2MASS'):
        if catalog == '2MASS':
            return self.match2MASS['{}+{}'.format(pupilshort,filtershort)]
        elif catalog == 'SIMBAD':
            return self.matchSIMBAD['{}+{}'.format(pupilshort,filtershort)]

class filter_info():
    '''
    Get info on filter wavelengths and bandpass
    '''    

    def __init__(self,filename= '/path/to/future/data_dir/' + 'Filter_info.txt'):

        self.filter_table = pd.read_csv(filename,sep='\s+')

    def get_info(self,pupilshort,filtershort,key_info='Pivot'):
        if pupilshort == 'CLEAR':
            check_value = filtershort
        else:
            check_value = pupilshort
        BM = self.filter_table['Filter'] == check_value
        return self.filter_table.loc[BM,key_info].values[0]
