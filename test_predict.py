from google.cloud import aiplatform as aip

from config import config
test_data = [{
    "date_decision": 1620950400000000000,
    "MONTH": 202201,
    "WEEK_NUM": 100,
    "actualdpdtolerance_344P": 0.0,
    "amtinstpaidbefduel24m_4187115A": 191767.36,
    "annuity_780A": 3674.60009765625,
    "annuitynextmonth_57A": 1218.2000732421875,
    "applicationcnt_361L": 0.0,
    "applications30d_658L": 0.0,
    "applicationscnt_1086L": 0.0,
    "applicationscnt_464L": 0.0,
    "applicationscnt_629L": 0.0,
    "applicationscnt_867L": 9.0,
    "avgdbddpdlast24m_3658932P": 1.0,
    "avgdbddpdlast3m_4187120P": 2.0,
    "avgdbdtollast24m_4525197P": 1.0,
    "avgdpdtolclosure24_3658938P": 1.0,
    "avginstallast24m_3658937A": 16049.400390625,
    "avglnamtstart24m_4525187A": 17054.400390625,
    "avgmaxdpdlast9m_3716943P": 2.0,
    "avgoutstandbalancel6m_4187114A": 14554.4,
    "avgpmtlast12m_4525200A": 24482.0,
    "bankacctype_710L": "CA",
    "cardtype_51L": None,
    "clientscnt12m_3712952L": 0.0,
    "clientscnt3m_3712950L": 0.0,
    "clientscnt6m_3712949L": 0.0,
    "clientscnt_100L": 0.0,
    "clientscnt_1022L": 0.0,
    "clientscnt_1071L": 0.0,
    "clientscnt_1130L": 0.0,
    "clientscnt_136L": None,
    "clientscnt_157L": 0.0,
    "clientscnt_257L": 0.0,
    "clientscnt_304L": 0.0,
    "clientscnt_360L": 0.0,
    "clientscnt_493L": 0.0,
    "clientscnt_533L": 0.0,
    "clientscnt_887L": 0.0,
    "clientscnt_946L": 0.0,
    "cntincpaycont9m_3716944L": 5.0,
    "cntpmts24_3658933L": 20.0,
    "commnoinclast6m_3546845L": 0.0,
    "credamount_770A": 20000.0,
    "credtype_322L": "CAL",
    "currdebt_22A": 12154.4,
    "currdebtcredtyperange_828A": 0.0,
    "datefirstoffer_1144D": 1205971200000000000,
    "datelastinstal40dpd_247D": 1590624000000000000,
    "datelastunpaid_3546854D": 1620864000000000000,
    "daysoverduetolerancedd_3976961L": 8.0,
    "deferredmnthsnum_166L": None,
    "disbursedcredamount_1113A": 20000.0,
    "disbursementtype_67L": "GBA",
    "downpmt_116A": 0.0,
    "dtlastpmtallstes_4499206D": 1621036800000000000,
    "eir_270L": 0.3400000035762787,
    "equalitydataagreement_891L": None,
    "equalityempfrom_62L": None,
    "firstclxcampaign_1125D": 1488326400000000000,
    "firstdatedue_489D": 1327363200000000000,
    "homephncnt_628L": 0.0,
    "inittransactionamount_650A": None,
    "inittransactioncode_186L": "CASH",
    "interestrate_311L": 0.3400000035762787,
    "interestrategrace_34L": None,
    "isbidproduct_1095L": True,
    "isbidproductrequest_292L": None,
    "isdebitcard_729L": False,
    "lastactivateddate_801D": 1619395200000000000,
    "lastapplicationdate_877D": 1617408000000000000,
    "lastapprcommoditycat_1041M": "a55475b1",
    "lastapprcommoditytypec_5251766M": "a55475b1",
    "lastapprcredamount_781A": 14000.0,
    "lastapprdate_640D": 1617408000000000000,
    "lastcancelreason_561M": "a55475b1",
    "lastdelinqdate_224D": 1620864000000000000,
    "lastdependentsnum_448L": None,
    "lastotherinc_902A": None,
    "lastotherlnsexpense_631A": None,
    "lastrejectcommoditycat_161M": "P109_133_183",
    "lastrejectcommodtypec_5251769M": "P49_111_165",
    "lastrejectcredamount_222A": 24000.0,
    "lastrejectdate_50D": 1439510400000000000,
    "lastrejectreason_759M": "a55475b1",
    "lastrejectreasonclient_4145040M": "a55475b1",
    "lastrepayingdate_696D": -9223372036854775808,
    "lastst_736L": "K",
    "maininc_215A": 34000.0,
    "mastercontrelectronic_519L": 0.0,
    "mastercontrexist_109L": 0.0,
    "maxannuity_159A": 280983.56,
    "maxannuity_4075009A": None,
    "maxdbddpdlast1m_3658939P": 2.0,
    "maxdbddpdtollast12m_3658940P": 3.0,
    "maxdbddpdtollast6m_4187119P": 3.0,
    "maxdebt4_972A": 231440.03,
    "maxdpdfrom6mto36m_3546853P": 7.0,
    "maxdpdinstldate_3546855D": 1615593600000000000,
    "maxdpdinstlnum_3546846P": 14.0,
    "maxdpdlast12m_727P": 3.0,
    "maxdpdlast24m_143P": 7.0,
    "maxdpdlast3m_392P": 3.0,
    "maxdpdlast6m_474P": 3.0,
    "maxdpdlast9m_1059P": 3.0,
    "maxdpdtolerance_374P": 7.0,
    "maxinstallast24m_3658928A": 131700.8,
    "maxlnamtstart6m_4525199A": 16672.6,
    "maxoutstandbalancel12m_4187113A": 157731.78,
    "maxpmtlast3m_4525190A": 16641.4,
    "mindbddpdlast24m_3658935P": -7.0,
    "mindbdtollast24m_4525191P": -7.0,
    "mobilephncnt_593L": 2.0,
    "monthsannuity_845L": 66.0,
    "numactivecreds_622L": 1.0,
    "numactivecredschannel_414L": 0.0,
    "numactiverelcontr_750L": 0.0,
    "numcontrs3months_479L": 1.0,
    "numincomingpmts_3546848L": 112.0,
    "numinstlallpaidearly3d_817L": 34.0,
    "numinstls_657L": 14.0,
    "numinstlsallpaid_934L": 66.0,
    "numinstlswithdpd10_728L": 0.0,
    "numinstlswithdpd5_4187116L": 6.0,
    "numinstlswithoutdpd_562L": 79.0,
    "numinstmatpaidtearly2d_4499204L": 37.0,
    "numinstpaid_4499208L": 96.0,
    "numinstpaidearly3d_3546850L": 34.0,
    "numinstpaidearly3dest_4493216L": 34.0,
    "numinstpaidearly5d_1087L": 0.0,
    "numinstpaidearly5dest_4493211L": 0.0,
    "numinstpaidearly5dobd_4499205L": 25.0,
    "numinstpaidearly_338L": 25.0,
    "numinstpaidearlyest_4493214L": 25.0,
    "numinstpaidlastcontr_4325080L": 1.0,
    "numinstpaidlate1d_3546852L": 31.0,
    "numinstregularpaid_973L": 96.0,
    "numinstregularpaidest_4493210L": 96.0,
    "numinsttopaygr_769L": 10.0,
    "numinsttopaygrest_4493213L": 10.0,
    "numinstunpaidmax_3546851L": 10.0,
    "numinstunpaidmaxest_4493212L": 10.0,
    "numnotactivated_1143L": 0.0,
    "numpmtchanneldd_318L": 0.0,
    "numrejects9m_859L": 0.0,
    "opencred_647L": False,
    "paytype1st_925L": None,
    "paytype_783L": None,
    "payvacationpostpone_4187118D": 1590192000000000000,
    "pctinstlsallpaidearl3d_427L": 0.3541699945926666,
    "pctinstlsallpaidlat10d_839L": 0.0,
    "pctinstlsallpaidlate1d_3546856L": 0.3229199945926666,
    "pctinstlsallpaidlate4d_3546849L": 0.07292000204324722,
    "pctinstlsallpaidlate6d_3546844L": 0.052080001682043076,
    "pmtnum_254L": 6.0,
    "posfpd10lastmonth_333P": 0.0,
    "posfpd30lastmonth_3976960P": 0.0,
    "posfstqpd30lastmonth_3976962P": 0.0,
    "previouscontdistrict_112M": "a55475b1",
    "price_1097A": 0.0,
    "sellerplacecnt_915L": 0.0,
    "sellerplacescnt_216L": 5.0,
    "sumoutstandtotal_3546847A": 12154.4,
    "sumoutstandtotalest_4493215A": 12154.4,
    "totaldebt_9A": 12154.4,
    "totalsettled_863A": 456031.1,
    "totinstallast1m_4525188A": 17859.599609375,
    "twobodfilling_608L": "FO",
    "typesuite_864L": "AL",
    "validfrom_1069D": -9223372036854775808,
    "assignmentdate_238D": -9223372036854775808,
    "assignmentdate_4527235D": -9223372036854775808,
    "assignmentdate_4955616D": -9223372036854775808,
    "birthdate_574D": -9223372036854775808,
    "contractssum_5085716L": 151364.0,
    "dateofbirth_337D": 341884800000000000,
    "dateofbirth_342D": -9223372036854775808,
    "days120_123L": 2.0,
    "days180_256L": 4.0,
    "days30_165L": 1.0,
    "days360_512L": 8.0,
    "days90_310L": 2.0,
    "description_5085714M": "2fc785b2",
    "education_1103M": "6b2ae0fa",
    "education_88M": "a55475b1",
    "firstquarter_103L": 4.0,
    "for3years_128L": None,
    "for3years_504L": None,
    "for3years_584L": None,
    "formonth_118L": None,
    "formonth_206L": None,
    "formonth_535L": None,
    "forquarter_1017L": None,
    "forquarter_462L": None,
    "forquarter_634L": None,
    "fortoday_1092L": None,
    "forweek_1077L": None,
    "forweek_528L": None,
    "forweek_601L": None,
    "foryear_618L": None,
    "foryear_818L": None,
    "foryear_850L": None,
    "fourthquarter_440L": 9.0,
    "maritalst_385M": "38c061ee",
    "maritalst_893M": "a55475b1",
    "numberofqueries_373L": 8.0,
    "pmtaverage_3A": None,
    "pmtaverage_4527227A": None,
    "pmtaverage_4955615A": None,
    "pmtcount_4527229L": None,
    "pmtcount_4955617L": None,
    "pmtcount_693L": None,
    "pmtscount_423L": None,
    "pmtssum_45A": None,
    "requesttype_4525192L": None,
    "responsedate_1012D": -9223372036854775808,
    "responsedate_4527233D": -9223372036854775808,
    "responsedate_4917613D": 1622160000000000000,
    "riskassesment_302T": None,
    "riskassesment_940T": None,
    "secondquarter_766L": 2.0,
    "thirdquarter_1082L": 3.0
}]

aip.init(project=config.PROJECT_ID, location=config.REGION)
target_endpoint = None
for endpoint in aip.Endpoint.list(order_by="update_time desc"):
    if endpoint.display_name == config.ENDPOINT_DISPLAY_NAME:
        target_endpoint = endpoint

prediction = target_endpoint.predict(instances=test_data)

print("PREDICTION:", prediction[0])