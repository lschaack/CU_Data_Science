import pandas
import math
from sklearn import linear_model, feature_extraction
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from zipfile import ZipFile

def categorical_features(row):
    d = {}
    d["STATE"] = row[1]["STATE"]
    return d

def last_poll(full_data):
    """
    Create feature from last poll in each state
    """
    
    # Only care about republicans
    repub = full_data[full_data["PARTY"] == "Rep"]

    # Sort by date
    chron = repub.sort_values(by="DATE", ascending=True)

    # Only keep the last one
    dedupe = chron.drop_duplicates(subset="STATE", keep="last")

    # Remove national polls
    return dedupe[dedupe["STATE"] != "US"]
    
if __name__ == "__main__":
    z = ZipFile("./data.zip")
    z.extractall()

    # Read in the X data
    all_data = pandas.read_csv("./data/data.csv")

    # Remove non-states
    all_data = all_data[pandas.notnull(all_data["STATE"])]

    """
    # Code to split train/test data for gauging model accuracy
    test_states = ('SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY')
    train_data = all_data[all_data["STATE"] != test_states]
    test_data = all_data[all_data["STATE"] == test_states]

    # split between testing and training
    train_x = last_poll(train_data[train_data["TOPIC"] == '2012-president'])
    test_x = last_poll(test_data[test_data["TOPIC"] == '2012-president'])
    """
    train_x = last_poll(all_data[all_data["TOPIC"] == '2012-president'])
    test_x = last_poll(all_data[all_data["TOPIC"] == '2016-president'])
    train_x.set_index("STATE")  
    test_x.set_index("STATE")
    
    # Read in the Y data
    y_data = pandas.read_csv("../data/2012_pres.csv", sep=';')
    y_data = y_data[y_data["PARTY"] == "R"]
    y_data = y_data[pandas.notnull(y_data["GENERAL %"])]
    y_data["GENERAL %"] = [float(x.replace(",", ".").replace("%", ""))
                           for x in y_data["GENERAL %"]]
    y_data["STATE"] = y_data["STATE ABBREVIATION"]
    y_data.set_index("STATE")

    """
    Reading (but not yet adding) more data.
    Using multi-line comments just to more visibly separate this code from the rest.
    """
    fields = ["STATE", "2010_DENSITY"]
    pop_density = pandas.read_csv("./data/pop_density_abbv.csv", usecols=fields)
    pop_density.set_index("STATE")

    fields = ["STATE", "Advanced"]
    educ = pandas.read_csv("./data/educ_abbv.csv", usecols=fields)
    educ.set_index("STATE")

    fields = ["STATE", "GOP"]
    more_polls_all_train = pandas.read_csv("./data/pres_polls_2012_abbv.csv", usecols=fields)
    more_polls_all_test = pandas.read_csv("./data/pres_polls_2016_abbv.csv", usecols=fields)
    more_polls_train = more_polls_all_train.drop_duplicates(subset="STATE", keep="last")
    more_polls_test = more_polls_all_test.drop_duplicates(subset="STATE", keep="last")
    more_polls_train.set_index("STATE")
    more_polls_test.set_index("STATE")

    fields = ["STATE", "StateTaxRate"]
    tax_rates = pandas.read_csv("./data/tax_rates_abbv.csv", usecols=fields)
    tax_rates["StateTaxRate"] = [float(x.replace("%", "")) for x in tax_rates["StateTaxRate"]]
    tax_rates.set_index("STATE")

    fields = ["STATE", "% African-American"]
    afr_am_pop = pandas.read_csv("./data/afr_am_population.csv", sep=";", usecols=fields)
    afr_am_pop["% African-American"] = [float(x.replace("%", "")) for x in afr_am_pop["% African-American"]]
    afr_am_pop.set_index("STATE")
    """
    / reading more data
    """

    backup = train_x
    train_x = y_data.merge(train_x, on="STATE", how="left")
    
    # make sure we have all states in the test data
    for ii in set(y_data.STATE) - set(test_x.STATE):
        new_row = pandas.DataFrame([{"STATE": ii}])
        test_x = test_x.append(new_row)

    """
    adding new data now that merging into all states is taken care of
    """
    train_x = pop_density.merge(train_x, on="STATE", how="right")
    test_x = pop_density.merge(test_x, on="STATE", how="right")

    train_x = educ.merge(train_x, on="STATE", how="right")
    test_x = educ.merge(test_x, on="STATE", how="right")

    train_x = more_polls_train.merge(train_x, on="STATE", how="right")
    test_x = more_polls_test.merge(test_x, on="STATE", how="right")

    train_x = tax_rates.merge(train_x, on="STATE", how="right")
    test_x = tax_rates.merge(test_x, on="STATE", how="right")

    train_x = afr_am_pop.merge(train_x, on="STATE", how="right")
    test_x = afr_am_pop.merge(test_x, on="STATE", how="right")
    """
    / adding new data
    """

    # format the data for regression
    train_x = pandas.concat([train_x.STATE.astype(str).str.get_dummies(),
                             train_x], axis=1)
    test_x = pandas.concat([test_x.STATE.astype(str).str.get_dummies(),
                             test_x], axis=1)
        
    # handle missing data   
    for dd in train_x, test_x:                
        dd["NOPOLL"] = pandas.isnull(dd["VALUE"])
        dd["VALUE"] = dd["VALUE"].fillna(0.0)
        
    # create feature list
    features = list(y_data.STATE)
    features.append("VALUE")
    features.append("NOPOLL")
    features.append("2010_DENSITY") # Most recent measure of population density
    features.append("Advanced") # Percentage of state population with advanced degrees
    features.append("GOP") # GOP Vote share of additional polls
    features.append("StateTaxRate") # Standard or average income tax rate by state
    features.append("% African-American") # African-American percentage of population
        
    # fit the regression
    mod = Pipeline([('poly', PolynomialFeatures(degree=2)),
                    ('linear', LinearRegression())])
    mod.set_params(linear__normalize=True) # correct for odd ranges in feature data
    mod.fit(train_x[features], train_x["GENERAL %"])

    # Write out the model
    with open("model.txt", 'w') as out:
        params = mod.get_params()
        for key in params.keys():
            out.write(key + ': ' + str(params[key]) + '\n')
        """
        # Pipeline doesn't have these attributes
        out.write("BIAS\t%f\n" % mod.intercept_)
        for jj, kk in zip(features, mod.coef_):
            out.write("%s\t%f\n" % (jj, kk))
        """

    # Write the predictions
    pred_test = mod.predict(test_x[features])
    with open("pred.txt", 'w') as out:
        for ss, vv in sorted(zip(list(test_x.STATE), pred_test)):
            out.write("%s\t%f\n" % (ss, vv))
    
    """
    # Determine RMSE for split train/test data
    sample = {}
    with open("pred3.txt") as file:
        for line in file:
            pred = line.split()
            if pred[0] in test_states:
                sample[pred[0]] = pred[1]

    popul = {}
    for key in sample.keys():
        popul[key] = y_data[y_data["STATE"] == key]["GENERAL %"]

    rmse = 0
    for key in sample.keys():
        rmse += (float(sample[key]) - float(popul[key])) ** 2
    rmse = math.sqrt(rmse / len(sample.keys()))
    print(rmse)
    """
