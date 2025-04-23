# iccs2025-ABM-CFD

This repository contains code for the research paper "Estimating airborne transmission risk for indoor space: Coupling agent-based model and computational fluid dynamics."

Authors: Boon Leng Ang, Jaeyoung Kwak, Chin Chun Ooi, Zhengwei Ge, Hongying Li, Michael Lees and Wentong Cai


In this study, we developed a framework integrating agent-based modeling (ABM) and computational fluid dynamics (CFD) to evaluate the transmission risk systematically. The developed framework was applied to a preschool COVID-19 cluster in Singapore as a case study. 

Our pedestrian movement model is implemented in MomenTUMv2 based on the concept of hierarchical behavior modeling which describes the pedestrian behavior in terms of three interconnected layers: the strategic layer, the tactical layer and the operational layer. MomenTUMv2 software can be found from https://github.com/tumcms/MomenTUM. 

Along with the pedeatrian trajectory records, we computed the location-specific concentration of virus particles emitted by the contagious individual by means of computational fluid dynamics (CFD) simulations. Based on the work of Ooi et al. (see https://doi.org/10.1063/5.0055547), we numerically solved the Navier-Stokes equation for conservation of mass and momentum, and the energy equation using a computational fluid dynamics (CFD) software (ANSYS FLUENT version 21.2).
