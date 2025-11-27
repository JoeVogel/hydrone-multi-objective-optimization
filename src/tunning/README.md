

setwd("D:\\Mestrado\\Dissertação\\hydrone-multi-objective-optimization\\src\\tunning")

library("irace")

scenario <- readScenario("scenario_win.txt")

checkIraceScenario(scenario)

irace(scenario)