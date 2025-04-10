#Test Nets
#Current file containing all nets (may be split into separate files later)

import torch
import torch.nn as nn

print("torch.cuda.is_available() == ", torch.cuda.is_available()) #For sanity, May be removed/Commented out

#Recovery net, Inititally made for predicting Outcome Recovery but structure works for all categorical outcomes possibly RENAME
class recovery_net(nn.Module):
    def __init__(self):
        super(recovery_net, self).__init__()
        self.name = "recovery_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(768, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
                )

    def forward(self, x):

        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)

        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)

        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)

        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)

        output = self.smplReg(xin)

        return output


#BackPain net, For predicting Outcome BackPain
class backpain_net(nn.Module):
    def __init__(self):
        super(backpain_net, self).__init__()
        self.name = "backpain_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(768, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
                )

    def forward(self, x):

        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)

        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)

        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)

        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)

        output = self.smplReg(xin)

        return output


#LegPain net, For predicting Outcome LegPain
class legpain_net(nn.Module):
    def __init__(self):
        super(legpain_net, self).__init__()
        self.name = "legpain_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(768, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
                )

    def forward(self, x):

        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)

        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)

        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)

        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)

        output = self.smplReg(xin)

        return output


#EQ IdxTL12 net, For predicting Outcome ODI Score
class eqidxtl12_net(nn.Module):
    def __init__(self):
        super(eqidxtl12_net, self).__init__()
        self.name = "eqidxtl12_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(768, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
                )

    def forward(self, x):
        
        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)

        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)

        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)

        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)

        output = self.smplReg(xin)

        return output


#ODIScore net, For predicting Outcome ODI Score
class odiscore_net(nn.Module):
    def __init__(self):
        super(odiscore_net, self).__init__()
        self.name = "odiscore_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(768, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
                )

    def forward(self, x):

        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)

        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)

        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)

        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)

        output = self.smplReg(xin)

        return output


#ODI4_Final net, For predicting Outcome ODI4_Final
class odi4_net(nn.Module):
    def __init__(self):
        super(odi4_net, self).__init__()
        self.name = "odi4_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 768),
                nn.BatchNorm1d(768),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(768, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.ReLU()
                )
                
    def forward(self, x):

        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)

        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)

        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)

        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)


        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)

        output = self.smplReg(xin)

        return output


#Full net, For predicting All Outcomes
class Full_net(nn.Module):
    def __init__(self):
        super(Full_net, self).__init__()
        self.name = "Full_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.married_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.education_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.painmed_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.inflammatory_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.musclerelax_embed = nn.Sequential(nn.Embedding(3, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.EClegpain_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECindependence_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECsocial_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(4, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())
        self.physio_embed = nn.Sequential(nn.Embedding(2, 12), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12,1), nn.ReLU())

        self.smplReg = nn.Sequential(
                nn.Linear(36, 96),
                nn.ReLU(),
                nn.Dropout(p=0.001),
                nn.Linear(96, 96),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(96, 32),
                nn.ReLU(),
                )

        self.BackPainReg = nn.Sequential(
                nn.Linear(32, 48),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(48, 1),
                nn.ReLU()
                )

        self.LegPainReg = nn.Sequential(
                nn.Linear(32, 48),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(48, 1),
                nn.ReLU()
                )

        self.ODIScoreReg = nn.Sequential(
                nn.Linear(32, 48),
                nn.BatchNorm1d(48),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(48, 1),
                nn.ReLU()
                )

        self.ODI4FinalReg = nn.Sequential(
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(32, 1),
                nn.ReLU()
                )

        self.EQIndexTL12Reg = nn.Sequential(
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(16, 1),
                nn.ReLU()
                )

        self.RecoveryReg = nn.Sequential(
                nn.Linear(32, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(32, 1),
                nn.Sigmoid()
                )

    def forward(self, x):

        xcon = x[:,:7].to(torch.float32)

        xcat = x[:,7:]

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        xsx = self.sx_embed(xsx).squeeze(2)

        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)

        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        xmarried = self.married_embed(xmarried).squeeze(2)

        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        xeducation = self.education_embed(xeducation).squeeze(2)

        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)

        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        xexercise = self.exercise_embed(xexercise).squeeze(2)

        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)

        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)

        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)

        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)

        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)

        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)

        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)

        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)

        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)

        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)
        
        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)

        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)
        
        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)

        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)

        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)

        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)

        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)
        
        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)

        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)

        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)

        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)

        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        xchiro = self.chiro_embed(xchiro).squeeze(2)

        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        
        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)

        xin = torch.cat((xcon,xcatemb), 1)
        output = self.smplReg(xin)

        outBackPain = self.BackPainReg(output)
        outLegPain = self.LegPainReg(output)
        outODIScore = self.ODIScoreReg(output)
        outODI4Final = self.ODI4FinalReg(output)
        outEQIndexTL12 = self.EQIndexTL12Reg(output)
        outRecovery = self.RecoveryReg(output)

        ret = torch.cat((outBackPain,outLegPain,outODIScore,outODI4Final,outEQIndexTL12,outRecovery), 1)

        return ret