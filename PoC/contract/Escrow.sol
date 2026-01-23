// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Escrow {
    struct Job {
        address client;
        address[3] parties;
        uint256 deposit;
        bool completed;
    }

    mapping(bytes32 => Job) public jobs;

    event JobCreated(
        bytes32 indexed jobId,
        address client,
        address[3] parties,
        uint256 deposit
    );
    event JobCompleted(bytes32 indexed jobId);

    function createJob(
        bytes32 jobId,
        address[3] calldata parties
    ) external payable {
        require(jobs[jobId].client == address(0), "Job already exists");
        require(msg.value > 0, "Deposit required");

        jobs[jobId] = Job({
            client: msg.sender,
            parties: parties,
            deposit: msg.value,
            completed: false
        });

        emit JobCreated(jobId, msg.sender, parties, msg.value);
    }

    function completeJob(
        bytes32 jobId,
        bytes calldata sig0,
        bytes calldata sig1,
        bytes calldata sig2
    ) external {
        Job storage job = jobs[jobId];
        require(job.client != address(0), "Job does not exist");
        require(!job.completed, "Job already completed");
        require(msg.sender == job.client, "Only client can complete");

        // Verify signatures
        bytes32 message = keccak256(abi.encodePacked(jobId, job.client));
        require(verifySignature(message, sig0, job.parties[0]), "Invalid sig0");
        require(verifySignature(message, sig1, job.parties[1]), "Invalid sig1");
        require(verifySignature(message, sig2, job.parties[2]), "Invalid sig2");

        // Split deposit equally
        uint256 share = job.deposit / 3;
        payable(job.parties[0]).transfer(share);
        payable(job.parties[1]).transfer(share);
        payable(job.parties[2]).transfer(share);

        job.completed = true;
        emit JobCompleted(jobId);
    }

    function verifySignature(
        bytes32 message,
        bytes memory sig,
        address signer
    ) internal pure returns (bool) {
        bytes32 ethMessage = keccak256(
            abi.encodePacked("\x19Ethereum Signed Message:\n32", message)
        );
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(sig);
        address recovered = ecrecover(ethMessage, v, r, s);
        return recovered == signer;
    }

    function splitSignature(
        bytes memory sig
    ) internal pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "Invalid signature length");
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
    }
}
