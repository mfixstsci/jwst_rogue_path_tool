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
