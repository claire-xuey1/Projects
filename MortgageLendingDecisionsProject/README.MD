# Background

The project is based on the paper and dataset in Alicia H. Munnell, Geoffrey M.B. Tootell, Lynn E. Browne, and James McEneaney (1996), “Mortgage Lending in Boston: Interpreting HMDA Data,” American Economic Review 86, 25-53.
The Home Mortgage Disclosure Act was enacted to monitor minority and low-income access to the mortgage market. The data collected for this purpose show that minorities are more than twice as likely to be denied a mortgage as whites.  Yet variables correlated with both race and creditworthiness were omitted from these data, making any conclusion about race’s role in mortgage lending impossible. The Federal Reserve Board of Boston collected additional variables important to the mortgage lending decision….
As discussed in Munnell et al (1996), the HMDA data indicate whether an applicant’s mortgage application was approved and provide several demographic characteristics.  In 1990, following the request of the Federal Reserve Board of Boston, lending institutions in the Boston area provided additional information relevant to mortgage lending decisions.  In light of the relatively small number of mortgage loan applications made by minorities, these extra variables were collected for all applications by blacks and Hispanics and for a random sample of those by whites.


The data set includes the following variables:
APPROVE = 1 if mortgage loan was approved, = 0 otherwise
GDLIN = 1 if credit history meets guidelines, = 0 otherwise
LOANPRC = loan amount/purchase price
OBRAT = other obligations as a percent of total income
MALE = 1 if male, = 0 otherwise
MARRIED = 1 if married, = 0 otherwise
BLACK = 1 if black, = 0 otherwise
HISPAN = 1 if Hispanic, = 0 otherwise
(All applicants are non-Hispanic white, non Hispanic black, or Hispanic.  In 1990 about 94% of Boston residents were white, Black, or Hispanic. (Links to an external site.))

In the project, we apply probit and logit models to predit probabilities of loan approval for the entire sample and for each racial/ethnic group.
After controlling for relevant characteristics, we analyze the race/ethnicity effect associated with the outcome of a mortgage loan application.
