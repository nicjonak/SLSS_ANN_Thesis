#Test Nets
#Current file containing all nets (may be split into separate files later)

import torch
import torch.nn as nn

print("torch.cuda.is_available() == ", torch.cuda.is_available())

#Simple net for use on continous outcomes
class smplNetCnt(nn.Module):
    def __init__(self):
        super(smplNetCnt, self).__init__()
        self.name = "smplNetCnt"
        self.smplLinRelu = nn.Sequential(
                nn.Linear(36, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                #nn.Sigmoid()
                nn.ReLU()
                )

    def forward(self, x):
        output = self.smplLinRelu(x.to(torch.float32))
        #print(" In Net: output = ", output)
        #output = torch.mul(output,10)
        #output = torch.argmax(output, dim=1).unsqueeze(1)
        #print(" In Net: After Mul: output = ", output)
        #output = torch.round(output)
        #print(" In Net: After Round: output = ", output)
        return output

#Simple net for use on categorical outcomes
class smplNetLog(nn.Module):
    def __init__(self):
        super(smplNetLog, self).__init__()
        self.name = "smplNetLog"
        self.smplLinRelu = nn.Sequential(
                nn.Linear(36, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
                #nn.ReLU()
                #nn.Sigmoid()
                nn.Softmax(dim=1)
                )

    def forward(self, x):
        output = self.smplLinRelu(x.to(torch.float32))
        #print(" In Net: output = ", output)
        #output = torch.mul(output,10)
        #output = torch.argmax(output, dim=1).unsqueeze(1)
        #print(" In Net: After Mul: output = ", output)
        #output = torch.round(output)
        #print(" In Net: After Round: output = ", output)
        return output

#Recovery net, Inititally made for predicting Outcome Recovery but structure works for all categorical outcomes possibly RENAME
class recovery_net(nn.Module):
    def __init__(self):
        super(recovery_net, self).__init__()
        self.name = "recovery_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.married_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.education_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.painmed_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.inflammatory_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.musclerelax_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.EClegpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECindependence_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsocial_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physio_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))

        self.catdo_layer = nn.Dropout(p=0.05)
        self.conbn_layer = nn.BatchNorm1d(7)

        self.smplReg = nn.Sequential(
                nn.Linear(36, 36),
                nn.BatchNorm1d(36),
                nn.ReLU(),
                nn.Linear(36, 18),
                nn.BatchNorm1d(18),
                nn.ReLU(),
                nn.Dropout(p=0.001),
                nn.Linear(18, 9),
                nn.BatchNorm1d(9),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(9, 2),
                nn.Softmax(dim=1)
                )
        

    def forward(self, x):
        #print("x",x)

        xcon = x[:,:7].to(torch.float32)
        #print("xcon", xcon)
        #print(" xcon.size = ", xcon.size())

        xcat = x[:,7:]
        #print("xcat", xcat)
        #print(" xcat.size = ", xcat.size())

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        #print(" xsx = ", xsx)
        xsx = self.sx_embed(xsx).squeeze(2)
        #print(" after embed xsx.size = ", xsx.size())
        #print(" after embed xsx = ", xsx)
        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        #print(" xsympdurat = ", xsympdurat)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)
        #print(" after embed xsympdurat.size = ", xsympdurat.size())
        #print(" after embed xsympdurat = ", xsympdurat)
        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        #print(" xmarried = ", xmarried)
        xmarried = self.married_embed(xmarried).squeeze(2)
        #print(" after embed xmarried.size = ", xmarried.size())
        #print(" after embed xmarried = ", xmarried)
        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        #print(" xeducation = ", xeducation)
        xeducation = self.education_embed(xeducation).squeeze(2)
        #print(" after embed xeducation.size = ", xeducation.size())
        #print(" after embed xeducation = ", xeducation)
        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        #print(" xsmoke = ", xsmoke)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)
        #print(" after embed xsmoke.size = ", xsmoke.size())
        #print(" after embed xsmoke = ", xsmoke)
        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        #print(" xexercise = ", xexercise)
        xexercise = self.exercise_embed(xexercise).squeeze(2)
        #print(" after embed xexercise.size = ", xexercise.size())
        #print(" after embed xexercise = ", xexercise)
        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        #print(" xworkstat = ", xworkstat)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)
        #print(" after embed xworkstat.size = ", xworkstat.size())
        #print(" after embed xworkstat = ", xworkstat)
        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        #print(" xchiroN = ", xchiroN)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)
        #print(" after embed xchiroN.size = ", xchiroN.size())
        #print(" after embed xchiroN = ", xchiroN)
        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        #print(" xphysioN = ", xphysioN)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)
        #print(" after embed xphysioN.size = ", xphysioN.size())
        #print(" after embed xphysioN = ", xphysioN)
        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        #print(" xtrainer = ", xtrainer)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)
        #print(" after embed xtrainer.size = ", xtrainer.size())
        #print(" after embed xtrainer = ", xtrainer)
        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        #print(" xpainmed = ", xpainmed)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)
        #print(" after embed xpainmed.size = ", xpainmed.size())
        #print(" after embed xpainmed = ", xpainmed)
        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        #print(" xinflammatory = ", xinflammatory)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)
        #print(" after embed xinflammatory.size = ", xinflammatory.size())
        #print(" after embed xinflammatory = ", xinflammatory)
        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        #print(" xmusclerelax = ", xmusclerelax)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)
        #print(" after embed xmusclerelax.size = ", xmusclerelax.size())
        #print(" after embed xmusclerelax = ", xmusclerelax)
        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        #print(" xECbackpain = ", xECbackpain)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)
        #print(" after embed xECbackpain.size = ", xECbackpain.size())
        #print(" after embed xECbackpain = ", xECbackpain)
        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        #print(" xEClegpain = ", xEClegpain)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)
        #print(" after embed xEClegpain.size = ", xEClegpain.size())
        #print(" after embed xEClegpain = ", xEClegpain)
        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        #print(" xECindependence = ", xECindependence)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)
        #print(" after embed xECindependence.size = ", xECindependence.size())
        #print(" after embed xECindependence = ", xECindependence)
        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        #print(" xECsportsac = ", xECsportsac)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)
        #print(" after embed xECsportsac.size = ", xECsportsac.size())
        #print(" after embed xECsportsac = ", xECsportsac)
        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        #print(" xECphyscapac = ", xECphyscapac)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)
        #print(" after embed xECphyscapac.size = ", xECphyscapac.size())
        #print(" after embed xECphyscapac = ", xECphyscapac)
        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        #print(" xECsocial = ", xECsocial)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)
        #print(" after embed xECsocial.size = ", xECsocial.size())
        #print(" after embed xECsocial = ", xECsocial)
        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        #print(" xECwellbeing = ", xECwellbeing)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)
        #print(" after embed xECwellbeing.size = ", xECwellbeing.size())
        #print(" after embed xECwellbeing = ", xECwellbeing)
        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        #print(" xexpbackpain = ", xexpbackpain)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)
        #print(" after embed xexpbackpain.size = ", xexpbackpain.size())
        #print(" after embed xexpbackpain = ", xexpbackpain)
        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        #print(" xexplegpain = ", xexplegpain)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)
        #print(" after embed xexplegpain.size = ", xexplegpain.size())
        #print(" after embed xexplegpain = ", xexplegpain)
        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        #print(" xexpindependence = ", xexpindependence)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)
        #print(" after embed xexpindependence.size = ", xexpindependence.size())
        #print(" after embed xexpindependence = ", xexpindependence)
        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        #print(" xexpsports = ", xexpsports)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)
        #print(" after embed xexpsports.size = ", xexpsports.size())
        #print(" after embed xexpsports = ", xexpsports)
        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        #print(" xexpphyscap = ", xexpphyscap)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)
        #print(" after embed xexpphyscap.size = ", xexpphyscap.size())
        #print(" after embed xexpphyscap = ", xexpphyscap)
        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        #print(" xexpsocial = ", xexpsocial)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)
        #print(" after embed xexpsocial.size = ", xexpsocial.size())
        #print(" after embed xexpsocial = ", xexpsocial)
        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        #print(" xexpwellbeing = ", xexpwellbeing)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)
        #print(" after embed xexpwellbeing.size = ", xexpwellbeing.size())
        #print(" after embed xexpwellbeing = ", xexpwellbeing)
        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        #print(" xchiro = ", xchiro)
        xchiro = self.chiro_embed(xchiro).squeeze(2)
        #print(" after embed xchiro.size = ", xchiro.size())
        #print(" after embed xchiro = ", xchiro)
        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        #print(" xphysio = ", xphysio)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        #print(" after embed xphysio.size = ", xphysio.size())
        #print(" after embed xphysio = ", xphysio)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcatemb = self.catdo_layer(xcatemb)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcon = self.conbn_layer(xcon)

        #print("xcatemb.size = ", xcatemb.size())
        #print("xcon.size = ", xcon.size())

        xin = torch.cat((xcon,xcatemb), 1)

        #print("xin.size = ", xin.size())
        #print("xin = ", xin)

        output = self.smplReg(xin)

        #print("output.size = ", output.size())
        #print("output = ", output)

        return output





#BackPain net, For predicting Outcome BackPain
class backpain_net(nn.Module):
    def __init__(self):
        super(backpain_net, self).__init__()
        self.name = "backpain_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.married_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.education_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.painmed_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.inflammatory_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.musclerelax_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.EClegpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECindependence_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsocial_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physio_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))

        self.catdo_layer = nn.Dropout(p=0.05)
        self.conbn_layer = nn.BatchNorm1d(7)

        self.smplReg = nn.Sequential(
                nn.Linear(36, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0005),
                nn.Linear(64, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout(p=0.001),
                nn.Linear(40, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(16, 1),
                nn.Sigmoid()
                )
        

    def forward(self, x):
        #print("x",x)

        xcon = x[:,:7].to(torch.float32)
        #print("xcon", xcon)
        #print(" xcon.size = ", xcon.size())

        xcat = x[:,7:]
        #print("xcat", xcat)
        #print(" xcat.size = ", xcat.size())

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        #print(" xsx = ", xsx)
        xsx = self.sx_embed(xsx).squeeze(2)
        #print(" after embed xsx.size = ", xsx.size())
        #print(" after embed xsx = ", xsx)
        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        #print(" xsympdurat = ", xsympdurat)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)
        #print(" after embed xsympdurat.size = ", xsympdurat.size())
        #print(" after embed xsympdurat = ", xsympdurat)
        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        #print(" xmarried = ", xmarried)
        xmarried = self.married_embed(xmarried).squeeze(2)
        #print(" after embed xmarried.size = ", xmarried.size())
        #print(" after embed xmarried = ", xmarried)
        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        #print(" xeducation = ", xeducation)
        xeducation = self.education_embed(xeducation).squeeze(2)
        #print(" after embed xeducation.size = ", xeducation.size())
        #print(" after embed xeducation = ", xeducation)
        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        #print(" xsmoke = ", xsmoke)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)
        #print(" after embed xsmoke.size = ", xsmoke.size())
        #print(" after embed xsmoke = ", xsmoke)
        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        #print(" xexercise = ", xexercise)
        xexercise = self.exercise_embed(xexercise).squeeze(2)
        #print(" after embed xexercise.size = ", xexercise.size())
        #print(" after embed xexercise = ", xexercise)
        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        #print(" xworkstat = ", xworkstat)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)
        #print(" after embed xworkstat.size = ", xworkstat.size())
        #print(" after embed xworkstat = ", xworkstat)
        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        #print(" xchiroN = ", xchiroN)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)
        #print(" after embed xchiroN.size = ", xchiroN.size())
        #print(" after embed xchiroN = ", xchiroN)
        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        #print(" xphysioN = ", xphysioN)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)
        #print(" after embed xphysioN.size = ", xphysioN.size())
        #print(" after embed xphysioN = ", xphysioN)
        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        #print(" xtrainer = ", xtrainer)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)
        #print(" after embed xtrainer.size = ", xtrainer.size())
        #print(" after embed xtrainer = ", xtrainer)
        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        #print(" xpainmed = ", xpainmed)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)
        #print(" after embed xpainmed.size = ", xpainmed.size())
        #print(" after embed xpainmed = ", xpainmed)
        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        #print(" xinflammatory = ", xinflammatory)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)
        #print(" after embed xinflammatory.size = ", xinflammatory.size())
        #print(" after embed xinflammatory = ", xinflammatory)
        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        #print(" xmusclerelax = ", xmusclerelax)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)
        #print(" after embed xmusclerelax.size = ", xmusclerelax.size())
        #print(" after embed xmusclerelax = ", xmusclerelax)
        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        #print(" xECbackpain = ", xECbackpain)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)
        #print(" after embed xECbackpain.size = ", xECbackpain.size())
        #print(" after embed xECbackpain = ", xECbackpain)
        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        #print(" xEClegpain = ", xEClegpain)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)
        #print(" after embed xEClegpain.size = ", xEClegpain.size())
        #print(" after embed xEClegpain = ", xEClegpain)
        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        #print(" xECindependence = ", xECindependence)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)
        #print(" after embed xECindependence.size = ", xECindependence.size())
        #print(" after embed xECindependence = ", xECindependence)
        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        #print(" xECsportsac = ", xECsportsac)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)
        #print(" after embed xECsportsac.size = ", xECsportsac.size())
        #print(" after embed xECsportsac = ", xECsportsac)
        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        #print(" xECphyscapac = ", xECphyscapac)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)
        #print(" after embed xECphyscapac.size = ", xECphyscapac.size())
        #print(" after embed xECphyscapac = ", xECphyscapac)
        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        #print(" xECsocial = ", xECsocial)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)
        #print(" after embed xECsocial.size = ", xECsocial.size())
        #print(" after embed xECsocial = ", xECsocial)
        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        #print(" xECwellbeing = ", xECwellbeing)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)
        #print(" after embed xECwellbeing.size = ", xECwellbeing.size())
        #print(" after embed xECwellbeing = ", xECwellbeing)
        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        #print(" xexpbackpain = ", xexpbackpain)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)
        #print(" after embed xexpbackpain.size = ", xexpbackpain.size())
        #print(" after embed xexpbackpain = ", xexpbackpain)
        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        #print(" xexplegpain = ", xexplegpain)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)
        #print(" after embed xexplegpain.size = ", xexplegpain.size())
        #print(" after embed xexplegpain = ", xexplegpain)
        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        #print(" xexpindependence = ", xexpindependence)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)
        #print(" after embed xexpindependence.size = ", xexpindependence.size())
        #print(" after embed xexpindependence = ", xexpindependence)
        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        #print(" xexpsports = ", xexpsports)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)
        #print(" after embed xexpsports.size = ", xexpsports.size())
        #print(" after embed xexpsports = ", xexpsports)
        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        #print(" xexpphyscap = ", xexpphyscap)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)
        #print(" after embed xexpphyscap.size = ", xexpphyscap.size())
        #print(" after embed xexpphyscap = ", xexpphyscap)
        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        #print(" xexpsocial = ", xexpsocial)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)
        #print(" after embed xexpsocial.size = ", xexpsocial.size())
        #print(" after embed xexpsocial = ", xexpsocial)
        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        #print(" xexpwellbeing = ", xexpwellbeing)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)
        #print(" after embed xexpwellbeing.size = ", xexpwellbeing.size())
        #print(" after embed xexpwellbeing = ", xexpwellbeing)
        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        #print(" xchiro = ", xchiro)
        xchiro = self.chiro_embed(xchiro).squeeze(2)
        #print(" after embed xchiro.size = ", xchiro.size())
        #print(" after embed xchiro = ", xchiro)
        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        #print(" xphysio = ", xphysio)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        #print(" after embed xphysio.size = ", xphysio.size())
        #print(" after embed xphysio = ", xphysio)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcatemb = self.catdo_layer(xcatemb)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcon = self.conbn_layer(xcon)

        #print("xcatemb.size = ", xcatemb.size())
        #print("xcon.size = ", xcon.size())

        xin = torch.cat((xcon,xcatemb), 1)

        #print("xin.size = ", xin.size())
        #print("xin = ", xin)

        output = self.smplReg(xin)

        output = torch.mul(output, 10)
        #output = torch.round(output)

        #print("output.size = ", output.size())
        #print("output = ", output)

        return output












#EQ IdxTL12 net, For predicting Outcome ODI Score
class eqidxtl12_net(nn.Module):
    def __init__(self):
        super(eqidxtl12_net, self).__init__()
        self.name = "eqidxtl12_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.married_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.education_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.painmed_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.inflammatory_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.musclerelax_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.EClegpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECindependence_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsocial_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physio_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))

        self.catdo_layer = nn.Dropout(p=0.05)
        self.conbn_layer = nn.BatchNorm1d(7)

        self.smplReg = nn.Sequential(
                nn.Linear(36, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0005),
                nn.Linear(64, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout(p=0.001),
                nn.Linear(40, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(16, 1),
                nn.Sigmoid()
                )
        

    def forward(self, x):
        #print("x",x)

        xcon = x[:,:7].to(torch.float32)
        #print("xcon", xcon)
        #print(" xcon.size = ", xcon.size())

        xcat = x[:,7:]
        #print("xcat", xcat)
        #print(" xcat.size = ", xcat.size())

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        #print(" xsx = ", xsx)
        xsx = self.sx_embed(xsx).squeeze(2)
        #print(" after embed xsx.size = ", xsx.size())
        #print(" after embed xsx = ", xsx)
        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        #print(" xsympdurat = ", xsympdurat)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)
        #print(" after embed xsympdurat.size = ", xsympdurat.size())
        #print(" after embed xsympdurat = ", xsympdurat)
        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        #print(" xmarried = ", xmarried)
        xmarried = self.married_embed(xmarried).squeeze(2)
        #print(" after embed xmarried.size = ", xmarried.size())
        #print(" after embed xmarried = ", xmarried)
        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        #print(" xeducation = ", xeducation)
        xeducation = self.education_embed(xeducation).squeeze(2)
        #print(" after embed xeducation.size = ", xeducation.size())
        #print(" after embed xeducation = ", xeducation)
        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        #print(" xsmoke = ", xsmoke)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)
        #print(" after embed xsmoke.size = ", xsmoke.size())
        #print(" after embed xsmoke = ", xsmoke)
        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        #print(" xexercise = ", xexercise)
        xexercise = self.exercise_embed(xexercise).squeeze(2)
        #print(" after embed xexercise.size = ", xexercise.size())
        #print(" after embed xexercise = ", xexercise)
        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        #print(" xworkstat = ", xworkstat)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)
        #print(" after embed xworkstat.size = ", xworkstat.size())
        #print(" after embed xworkstat = ", xworkstat)
        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        #print(" xchiroN = ", xchiroN)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)
        #print(" after embed xchiroN.size = ", xchiroN.size())
        #print(" after embed xchiroN = ", xchiroN)
        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        #print(" xphysioN = ", xphysioN)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)
        #print(" after embed xphysioN.size = ", xphysioN.size())
        #print(" after embed xphysioN = ", xphysioN)
        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        #print(" xtrainer = ", xtrainer)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)
        #print(" after embed xtrainer.size = ", xtrainer.size())
        #print(" after embed xtrainer = ", xtrainer)
        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        #print(" xpainmed = ", xpainmed)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)
        #print(" after embed xpainmed.size = ", xpainmed.size())
        #print(" after embed xpainmed = ", xpainmed)
        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        #print(" xinflammatory = ", xinflammatory)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)
        #print(" after embed xinflammatory.size = ", xinflammatory.size())
        #print(" after embed xinflammatory = ", xinflammatory)
        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        #print(" xmusclerelax = ", xmusclerelax)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)
        #print(" after embed xmusclerelax.size = ", xmusclerelax.size())
        #print(" after embed xmusclerelax = ", xmusclerelax)
        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        #print(" xECbackpain = ", xECbackpain)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)
        #print(" after embed xECbackpain.size = ", xECbackpain.size())
        #print(" after embed xECbackpain = ", xECbackpain)
        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        #print(" xEClegpain = ", xEClegpain)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)
        #print(" after embed xEClegpain.size = ", xEClegpain.size())
        #print(" after embed xEClegpain = ", xEClegpain)
        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        #print(" xECindependence = ", xECindependence)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)
        #print(" after embed xECindependence.size = ", xECindependence.size())
        #print(" after embed xECindependence = ", xECindependence)
        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        #print(" xECsportsac = ", xECsportsac)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)
        #print(" after embed xECsportsac.size = ", xECsportsac.size())
        #print(" after embed xECsportsac = ", xECsportsac)
        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        #print(" xECphyscapac = ", xECphyscapac)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)
        #print(" after embed xECphyscapac.size = ", xECphyscapac.size())
        #print(" after embed xECphyscapac = ", xECphyscapac)
        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        #print(" xECsocial = ", xECsocial)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)
        #print(" after embed xECsocial.size = ", xECsocial.size())
        #print(" after embed xECsocial = ", xECsocial)
        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        #print(" xECwellbeing = ", xECwellbeing)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)
        #print(" after embed xECwellbeing.size = ", xECwellbeing.size())
        #print(" after embed xECwellbeing = ", xECwellbeing)
        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        #print(" xexpbackpain = ", xexpbackpain)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)
        #print(" after embed xexpbackpain.size = ", xexpbackpain.size())
        #print(" after embed xexpbackpain = ", xexpbackpain)
        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        #print(" xexplegpain = ", xexplegpain)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)
        #print(" after embed xexplegpain.size = ", xexplegpain.size())
        #print(" after embed xexplegpain = ", xexplegpain)
        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        #print(" xexpindependence = ", xexpindependence)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)
        #print(" after embed xexpindependence.size = ", xexpindependence.size())
        #print(" after embed xexpindependence = ", xexpindependence)
        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        #print(" xexpsports = ", xexpsports)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)
        #print(" after embed xexpsports.size = ", xexpsports.size())
        #print(" after embed xexpsports = ", xexpsports)
        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        #print(" xexpphyscap = ", xexpphyscap)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)
        #print(" after embed xexpphyscap.size = ", xexpphyscap.size())
        #print(" after embed xexpphyscap = ", xexpphyscap)
        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        #print(" xexpsocial = ", xexpsocial)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)
        #print(" after embed xexpsocial.size = ", xexpsocial.size())
        #print(" after embed xexpsocial = ", xexpsocial)
        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        #print(" xexpwellbeing = ", xexpwellbeing)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)
        #print(" after embed xexpwellbeing.size = ", xexpwellbeing.size())
        #print(" after embed xexpwellbeing = ", xexpwellbeing)
        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        #print(" xchiro = ", xchiro)
        xchiro = self.chiro_embed(xchiro).squeeze(2)
        #print(" after embed xchiro.size = ", xchiro.size())
        #print(" after embed xchiro = ", xchiro)
        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        #print(" xphysio = ", xphysio)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        #print(" after embed xphysio.size = ", xphysio.size())
        #print(" after embed xphysio = ", xphysio)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcatemb = self.catdo_layer(xcatemb)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        #print("pre bn xcon = ", xcon)

        xcon = self.conbn_layer(xcon)

        #print("post bn xcon = ", xcon)

        #print("xcatemb.size = ", xcatemb.size())
        #print("xcon.size = ", xcon.size())

        xin = torch.cat((xcon,xcatemb), 1)

        #print("xin.size = ", xin.size())
        #print("xin = ", xin)

        #print("Gets Here pre smplReg")

        output = self.smplReg(xin)

        #print("Gets Here post smplReg")

        #output = torch.mul(output, 100)
        #output = torch.round(output)

        #print("output.size = ", output.size())
        #print("output = ", output)

        return output






































#ODIScore net, For predicting Outcome ODI Score
class odiscore_net(nn.Module):
    def __init__(self):
        super(odiscore_net, self).__init__()
        self.name = "odiscore_net"

        self.sx_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.sympdurat_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.married_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.education_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.smoke_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.exercise_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.workstat_embed = nn.Sequential(nn.Embedding(3, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiroN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physioN_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.trainer_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.painmed_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.inflammatory_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.musclerelax_embed = nn.Sequential(nn.Embedding(4, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECbackpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.EClegpain_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECindependence_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsportsac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECphyscapac_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECsocial_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.ECwellbeing_embed = nn.Sequential(nn.Embedding(5, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expbackpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.explegpain_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expindependence_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsports_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expphyscap_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expsocial_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.expwellbeing_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.chiro_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))
        self.physio_embed = nn.Sequential(nn.Embedding(2, 6), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6,1))

        self.catdo_layer = nn.Dropout(p=0.05)
        self.conbn_layer = nn.BatchNorm1d(7)

        self.smplReg = nn.Sequential(
                nn.Linear(36, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0001),
                nn.Linear(64, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(p=0.0005),
                nn.Linear(64, 40),
                nn.BatchNorm1d(40),
                nn.ReLU(),
                nn.Dropout(p=0.001),
                nn.Linear(40, 16),
                nn.BatchNorm1d(16),
                nn.ReLU(),
                nn.Dropout(p=0.01),
                nn.Linear(16, 1),
                nn.Sigmoid()
                )
        

    def forward(self, x):
        #print("x",x)

        xcon = x[:,:7].to(torch.float32)
        #print("xcon", xcon)
        #print(" xcon.size = ", xcon.size())

        xcat = x[:,7:]
        #print("xcat", xcat)
        #print(" xcat.size = ", xcat.size())

        xsx = x[:,7].unsqueeze(1).to(torch.int64)
        #print(" xsx = ", xsx)
        xsx = self.sx_embed(xsx).squeeze(2)
        #print(" after embed xsx.size = ", xsx.size())
        #print(" after embed xsx = ", xsx)
        xsympdurat = x[:,8].unsqueeze(1).to(torch.int64)
        #print(" xsympdurat = ", xsympdurat)
        xsympdurat = self.sympdurat_embed(xsympdurat).squeeze(2)
        #print(" after embed xsympdurat.size = ", xsympdurat.size())
        #print(" after embed xsympdurat = ", xsympdurat)
        xmarried = x[:,9].unsqueeze(1).to(torch.int64)
        #print(" xmarried = ", xmarried)
        xmarried = self.married_embed(xmarried).squeeze(2)
        #print(" after embed xmarried.size = ", xmarried.size())
        #print(" after embed xmarried = ", xmarried)
        xeducation = x[:,10].unsqueeze(1).to(torch.int64)
        #print(" xeducation = ", xeducation)
        xeducation = self.education_embed(xeducation).squeeze(2)
        #print(" after embed xeducation.size = ", xeducation.size())
        #print(" after embed xeducation = ", xeducation)
        xsmoke = x[:,11].unsqueeze(1).to(torch.int64)
        #print(" xsmoke = ", xsmoke)
        xsmoke = self.smoke_embed(xsmoke).squeeze(2)
        #print(" after embed xsmoke.size = ", xsmoke.size())
        #print(" after embed xsmoke = ", xsmoke)
        xexercise = x[:,12].unsqueeze(1).to(torch.int64)
        #print(" xexercise = ", xexercise)
        xexercise = self.exercise_embed(xexercise).squeeze(2)
        #print(" after embed xexercise.size = ", xexercise.size())
        #print(" after embed xexercise = ", xexercise)
        xworkstat = x[:,13].unsqueeze(1).to(torch.int64)
        #print(" xworkstat = ", xworkstat)
        xworkstat = self.workstat_embed(xworkstat).squeeze(2)
        #print(" after embed xworkstat.size = ", xworkstat.size())
        #print(" after embed xworkstat = ", xworkstat)
        xchiroN = x[:,14].unsqueeze(1).to(torch.int64)
        #print(" xchiroN = ", xchiroN)
        xchiroN = self.chiroN_embed(xchiroN).squeeze(2)
        #print(" after embed xchiroN.size = ", xchiroN.size())
        #print(" after embed xchiroN = ", xchiroN)
        xphysioN = x[:,15].unsqueeze(1).to(torch.int64)
        #print(" xphysioN = ", xphysioN)
        xphysioN = self.physioN_embed(xphysioN).squeeze(2)
        #print(" after embed xphysioN.size = ", xphysioN.size())
        #print(" after embed xphysioN = ", xphysioN)
        xtrainer = x[:,16].unsqueeze(1).to(torch.int64)
        #print(" xtrainer = ", xtrainer)
        xtrainer = self.trainer_embed(xtrainer).squeeze(2)
        #print(" after embed xtrainer.size = ", xtrainer.size())
        #print(" after embed xtrainer = ", xtrainer)
        xpainmed = x[:,17].unsqueeze(1).to(torch.int64)
        #print(" xpainmed = ", xpainmed)
        xpainmed = self.painmed_embed(xpainmed).squeeze(2)
        #print(" after embed xpainmed.size = ", xpainmed.size())
        #print(" after embed xpainmed = ", xpainmed)
        xinflammatory = x[:,18].unsqueeze(1).to(torch.int64)
        #print(" xinflammatory = ", xinflammatory)
        xinflammatory = self.inflammatory_embed(xinflammatory).squeeze(2)
        #print(" after embed xinflammatory.size = ", xinflammatory.size())
        #print(" after embed xinflammatory = ", xinflammatory)
        xmusclerelax = x[:,19].unsqueeze(1).to(torch.int64)
        #print(" xmusclerelax = ", xmusclerelax)
        xmusclerelax = self.musclerelax_embed(xmusclerelax).squeeze(2)
        #print(" after embed xmusclerelax.size = ", xmusclerelax.size())
        #print(" after embed xmusclerelax = ", xmusclerelax)
        xECbackpain = x[:,20].unsqueeze(1).to(torch.int64)
        #print(" xECbackpain = ", xECbackpain)
        xECbackpain = self.ECbackpain_embed(xECbackpain).squeeze(2)
        #print(" after embed xECbackpain.size = ", xECbackpain.size())
        #print(" after embed xECbackpain = ", xECbackpain)
        xEClegpain = x[:,21].unsqueeze(1).to(torch.int64)
        #print(" xEClegpain = ", xEClegpain)
        xEClegpain = self.EClegpain_embed(xEClegpain).squeeze(2)
        #print(" after embed xEClegpain.size = ", xEClegpain.size())
        #print(" after embed xEClegpain = ", xEClegpain)
        xECindependence = x[:,22].unsqueeze(1).to(torch.int64)
        #print(" xECindependence = ", xECindependence)
        xECindependence = self.ECindependence_embed(xECindependence).squeeze(2)
        #print(" after embed xECindependence.size = ", xECindependence.size())
        #print(" after embed xECindependence = ", xECindependence)
        xECsportsac = x[:,23].unsqueeze(1).to(torch.int64)
        #print(" xECsportsac = ", xECsportsac)
        xECsportsac = self.ECsportsac_embed(xECsportsac).squeeze(2)
        #print(" after embed xECsportsac.size = ", xECsportsac.size())
        #print(" after embed xECsportsac = ", xECsportsac)
        xECphyscapac = x[:,24].unsqueeze(1).to(torch.int64)
        #print(" xECphyscapac = ", xECphyscapac)
        xECphyscapac = self.ECphyscapac_embed(xECphyscapac).squeeze(2)
        #print(" after embed xECphyscapac.size = ", xECphyscapac.size())
        #print(" after embed xECphyscapac = ", xECphyscapac)
        xECsocial = x[:,25].unsqueeze(1).to(torch.int64)
        #print(" xECsocial = ", xECsocial)
        xECsocial = self.ECsocial_embed(xECsocial).squeeze(2)
        #print(" after embed xECsocial.size = ", xECsocial.size())
        #print(" after embed xECsocial = ", xECsocial)
        xECwellbeing = x[:,26].unsqueeze(1).to(torch.int64)
        #print(" xECwellbeing = ", xECwellbeing)
        xECwellbeing = self.ECwellbeing_embed(xECwellbeing).squeeze(2)
        #print(" after embed xECwellbeing.size = ", xECwellbeing.size())
        #print(" after embed xECwellbeing = ", xECwellbeing)
        xexpbackpain = x[:,27].unsqueeze(1).to(torch.int64)
        #print(" xexpbackpain = ", xexpbackpain)
        xexpbackpain = self.expbackpain_embed(xexpbackpain).squeeze(2)
        #print(" after embed xexpbackpain.size = ", xexpbackpain.size())
        #print(" after embed xexpbackpain = ", xexpbackpain)
        xexplegpain = x[:,28].unsqueeze(1).to(torch.int64)
        #print(" xexplegpain = ", xexplegpain)
        xexplegpain = self.explegpain_embed(xexplegpain).squeeze(2)
        #print(" after embed xexplegpain.size = ", xexplegpain.size())
        #print(" after embed xexplegpain = ", xexplegpain)
        xexpindependence = x[:,29].unsqueeze(1).to(torch.int64)
        #print(" xexpindependence = ", xexpindependence)
        xexpindependence = self.expindependence_embed(xexpindependence).squeeze(2)
        #print(" after embed xexpindependence.size = ", xexpindependence.size())
        #print(" after embed xexpindependence = ", xexpindependence)
        xexpsports = x[:,30].unsqueeze(1).to(torch.int64)
        #print(" xexpsports = ", xexpsports)
        xexpsports = self.expsports_embed(xexpsports).squeeze(2)
        #print(" after embed xexpsports.size = ", xexpsports.size())
        #print(" after embed xexpsports = ", xexpsports)
        xexpphyscap = x[:,31].unsqueeze(1).to(torch.int64)
        #print(" xexpphyscap = ", xexpphyscap)
        xexpphyscap = self.expphyscap_embed(xexpphyscap).squeeze(2)
        #print(" after embed xexpphyscap.size = ", xexpphyscap.size())
        #print(" after embed xexpphyscap = ", xexpphyscap)
        xexpsocial = x[:,32].unsqueeze(1).to(torch.int64)
        #print(" xexpsocial = ", xexpsocial)
        xexpsocial = self.expsocial_embed(xexpsocial).squeeze(2)
        #print(" after embed xexpsocial.size = ", xexpsocial.size())
        #print(" after embed xexpsocial = ", xexpsocial)
        xexpwellbeing = x[:,33].unsqueeze(1).to(torch.int64)
        #print(" xexpwellbeing = ", xexpwellbeing)
        xexpwellbeing = self.expwellbeing_embed(xexpwellbeing).squeeze(2)
        #print(" after embed xexpwellbeing.size = ", xexpwellbeing.size())
        #print(" after embed xexpwellbeing = ", xexpwellbeing)
        xchiro = x[:,34].unsqueeze(1).to(torch.int64)
        #print(" xchiro = ", xchiro)
        xchiro = self.chiro_embed(xchiro).squeeze(2)
        #print(" after embed xchiro.size = ", xchiro.size())
        #print(" after embed xchiro = ", xchiro)
        xphysio = x[:,35].unsqueeze(1).to(torch.int64)
        #print(" xphysio = ", xphysio)
        xphysio = self.physio_embed(xphysio).squeeze(2)
        #print(" after embed xphysio.size = ", xphysio.size())
        #print(" after embed xphysio = ", xphysio)
        

        xcatemb = torch.cat((xsx, xsympdurat, xmarried, xeducation, xsmoke, xexercise, xworkstat, xchiroN, xphysioN, xtrainer, xpainmed, xinflammatory, xmusclerelax, xECbackpain, xEClegpain, xECindependence, xECsportsac, xECphyscapac, xECsocial, xECwellbeing, xexpbackpain, xexplegpain, xexpindependence, xexpsports, xexpphyscap, xexpsocial, xexpwellbeing, xchiro, xphysio), 1)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcatemb = self.catdo_layer(xcatemb)
        #print("xcatemb.size = ", xcatemb.size())
        #print(" xcatemb = ", xcatemb)

        xcon = self.conbn_layer(xcon)

        #print("xcatemb.size = ", xcatemb.size())
        #print("xcon.size = ", xcon.size())

        xin = torch.cat((xcon,xcatemb), 1)

        #print("xin.size = ", xin.size())
        #print("xin = ", xin)

        output = self.smplReg(xin)

        output = torch.mul(output, 100)
        #output = torch.round(output)

        #print("output.size = ", output.size())
        #print("output = ", output)

        return output