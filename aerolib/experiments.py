import os
import pandas as pd 
from simple_term_menu import TerminalMenu
import shutup 
shutup.please()

class Damage_Classes():
    def __init__(self, path='../../data/experimemnts.csv'):
        # some constant values
        self.windspeed = [10, 20]
        self.excitation = ['1,00', '1,90']
        self.labels = [
                (0, "Healthy Probes"),
                (1, "Healthy Probes with Additional Mass"),
                (2, "5mm Crack"),
                (3, "10mm Crack"),
                (4, "15mm Crack"),
                (5, "20mm Crack"),
                ]
         
        # load the data from the excel sheet
        cols = [
            'Experiment Number',
            'Angle of attack [deg.]',
            'Heaving frequency in [Hz],  from motor excitations',
            'Wind speed [m/s]',
            'Crack length [mm]',
            'Concentrated mass [yes = 1, no = 0]',
            ]

        self.df = pd.read_csv(path, usecols=cols)
        
        # add a colum with the labels to the df
        labels = []
        for i, row in self.df.iterrows():
            if row['Concentrated mass [yes = 1, no = 0]']:
                labels.append(self.labels[1][0])
            elif row['Crack length [mm]'] == 0:
                labels.append(self.labels[0][0])
            elif row['Crack length [mm]'] == 5:
                labels.append(self.labels[2][0])
            elif row['Crack length [mm]'] == 10:
                labels.append(self.labels[3][0])
            elif row['Crack length [mm]'] == 15:
                labels.append(self.labels[4][0])
            else:
                labels.append(self.labels[5][0])
        # workaround: excelsheet is missing the data
        labels[47]=2

        self.df.insert(0, "Label", labels ,True)

        #print(f"Imported Shape: {self.df.shape}")
    
    def getDf(self): return self.df

    def ex2label(self, experiment):
        return self.df.loc[self.df['Experiment Number'] == experiment]['Label'].iloc[0]

    def ex2wind(self, experiment):
        return self.df.loc[self.df['Experiment Number'] == experiment]['Wind speed [m/s]'].iloc[0]

    def ex2excit(self, experiment):
        return self.df.loc[self.df['Experiment Number'] == experiment]['Heaving frequency in [Hz],  from motor excitations'].iloc[0]

    # experiment to desc
    def ex2desc(self, experiment):
        label = self.ex2label(experiment)
        for l, d in self.labels:
            if l == label:
                return d
        

    # label to experiments
    def label2exlist(self, label):
        return self.df.loc[df['Label'] == label]['Experiment Number'].tolist()

    # if an argument is none take all
    def filter(self, label, wind=None, excitation=None):
        #print(f"Label: {label} | wind: {wind} | exci: {excitation}")
        df = self.df
        # print(df['Wind speed [m/s]'].isin(self.windspeed))
        if wind == None and excitation == None:
            return df.loc[(df['Label'] == label)&(df['Wind speed [m/s]'].isin(self.windspeed))]['Experiment Number'].tolist()

        if wind == None:
            return df.loc[(df['Label'] == label)&(df['Wind speed [m/s]'].isin(self.windspeed))&(df['Heaving frequency in [Hz],  from motor excitations'] == excitation)]['Experiment Number'].tolist()

        if excitation == None:
            return df.loc[(df['Label'] == label)&(df['Wind speed [m/s]'] == wind)]['Experiment Number'].tolist()

        return df.loc[(df['Label'] == label)&(df['Wind speed [m/s]'] == wind)&(df['Heaving frequency in [Hz],  from motor excitations'] == excitation)]['Experiment Number'].tolist()


def chooseSensor(allSensors=True, sensorCount=36, debug=False):
    opt = [x for x in range(1, sensorCount+1)]
    if allSensors: opt.append("all")
    if debug: print(f"Possible Options: {opt}")
    optStr = map(str, opt)
    idx = TerminalMenu(optStr, title="Choose a Sensor:").show()
    print(f"Selected Sensor: {opt[idx]}")
    return opt[idx]

def chooseSetOfExperiments():
    dc = Damage_Classes()

    # choose label
    options = []
    for l, d in dc.labels:
        options.append(d)
    label = TerminalMenu(options, title="Choose Damage Class:").show()
    damage = dc.labels[label][0]
    desc = dc.labels[label][1]
    print(f"Selected Damage Class: {desc} with label {damage}")

    # choose wind
    opt = [None]
    for d in dc.windspeed:
        opt.append(d)
    optStr = map(str, opt)
    idx = TerminalMenu(optStr, title="Choose Windspeed in m/s:").show()
    windspeed = opt[idx]
    print(f"Selected Windspeed: {windspeed} m/s")

    # choose excitation
    opt = [None]
    for d in dc.excitation:
        opt.append(d)
    optStr = map(str, opt)
    idx = TerminalMenu(optStr, title="Choose Excitation in Hz:").show()
    excitation = opt[idx]
    print(f"Selected Excitation Frequency: {excitation} Hz")

    desc = dc.labels[label][1]
    experiments = dc.filter(damage, wind=windspeed, excitation=excitation)
    return experiments, label, desc, windspeed, excitation

if __name__ == "__main__":
    # dc = Damage_Classes()
    #
    # # iterating the columns
    # df = dc.getDf()
    # print("\nPresent Colums:")
    # for col in df.columns:
    #     print(col)
    #
    # print(f"\nExperiment: 25 to label: {dc.ex2label(25)}")
    # print(f"\nExperiment: 25 to desc: {dc.ex2desc(25)}")
    # print(f"\nLabel: 2 to ExList: {dc.label2exlist(2)}")
    # print(f"\nLabel: 4, excit: 1,00, wind: all to ExList: {dc.filter(2, excitation='1,00')}")
    # print(f"\nLabel: 4, excit: 1,00, wind: all to ExList: {dc.filter(2, excitation='1,00')}")
    # print(f"\nLabel: 4, excit: 1,00, wind: 10 to ExList: {dc.filter(2, wind=10, excitation='1,00')}")
    # print(f"\nLabel: 4, excit: 1,90, wind: 20 to ExList: {dc.filter(2, wind=20, excitation='1,90')}")


    chooseSensor(debug=True)
    chooseSetOfExperiments()

