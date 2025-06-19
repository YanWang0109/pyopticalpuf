import numpy as np
from warnings import warn
from itertools import product
from tqdm import tqdm
from nistrng import *
from colorama import Fore, init, Style

class NIST:
    @staticmethod
    def testForDuplicateSegments(binarySequence: list, segmentLength: int = 1000) -> bool:
        init()
        numberOfBits = len(binarySequence)
        numberOfBitSegments = numberOfBits // segmentLength
        remainderBits = numberOfBits % segmentLength

        if remainderBits > 0:
            warn(f"Unable to split binary sequence of {numberOfBits} bits into exactly segments {segmentLength} long. The last {remainderBits} have been automatically discarded", RuntimeWarning)

        segments = np.array_split(binarySequence, numberOfBitSegments)
        allSegmentCombinations = list(product(segments, segments))

        allSegmentCombinationsWithoutDiplicates = [(segmentOne, segmentTwo) for segmentOne, segmentTwo in tqdm(allSegmentCombinations, "Splitting into segments") if (segmentOne is not segmentTwo) and (segmentOne.shape == segmentTwo.shape)]

        comparisonResults = [np.all(np.equal(segmentOne, segmentTwo)) for segmentOne, segmentTwo in tqdm(allSegmentCombinationsWithoutDiplicates, "Comparing every combination of segment")]

        indexOfDuplicateSegments = np.where(comparisonResults == np.True_)[0]

        if indexOfDuplicateSegments.size > 0:
            print(f"{indexOfDuplicateSegments.size} duplicate segments found")
            return False
        else:
            print(f"{Fore.GREEN}No duplicate segments found.{Style.RESET_ALL}")
            return True

    @staticmethod
    def runNISTTests(binary_sequence):
        """
        Run NIST statistical tests on the binary sequence.

        Parameters:
        - binary_sequence (numpy.ndarray): Binary sequence as a numpy array.
        """
        init()

        # Pack sequence for NIST tests
        packedBinarySequence = pack_sequence(binary_sequence)
        
        # Check eligibility of the test and generate eligible battery
        eligibleBattery = check_eligibility_all_battery(packedBinarySequence, SP800_22R1A_BATTERY)
        
        listOfAllChecks = list(eligibleBattery.keys())
        template = '{:<30} {:<30} {:<30} {:<30}'
        headers = ["\nTest name", "Result", "Score", "Time Taken (ms)"]
        print(template.format(*headers))

        testResults = {}
        for testName in listOfAllChecks:
            try:
                print(f"Calculating {testName}...", end='\r')
                testResult, timeTaken = run_by_name_battery(testName, packedBinarySequence.copy(), eligibleBattery, False)
                print(template.format(*[testResult.name, f"{Fore.GREEN}Passed{Style.RESET_ALL}" if testResult.passed else f"{Fore.RED}Failed{Style.RESET_ALL}", round(testResult.score, 3), timeTaken]))
                testResult.timeTaken = timeTaken
                testResults[testResult.name] = testResult
            except:
                print(template.format(*[testResult.name, f"{Fore.YELLOW}Error{Style.RESET_ALL}", " - ", " - "]))        
            
        return testResults