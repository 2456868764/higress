// Copyright (c) 2022 Alibaba Group Holding Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import "strings"

type Protocol string

const (
	TCP         Protocol = "TCP"
	HTTP        Protocol = "HTTP"
	HTTP2       Protocol = "HTTP2"
	HTTPS       Protocol = "HTTPS"
	GRPC        Protocol = "GRPC"
	GRPCS       Protocol = "GRPCS"
	Dubbo       Protocol = "Dubbo"
	Unsupported Protocol = "UnsupportedProtocol"
)

func ParseProtocol(s string) Protocol {
	switch strings.ToLower(s) {
	case "tcp":
		return TCP
	case "http":
		return HTTP
	case "https":
		return HTTPS
	case "http2":
		return HTTP2
	case "grpc", "triple", "tri":
		return GRPC
	case "grpcs":
		return GRPCS
	case "dubbo":
		return Dubbo
	}
	return Unsupported
}

func (p Protocol) IsTCP() bool {
	switch p {
	case TCP:
		return true
	default:
		return false
	}
}

func (p Protocol) IsHTTP() bool {
	switch p {
	case HTTP, GRPC, GRPCS, HTTP2, HTTPS:
		return true
	default:
		return false
	}
}

func (p Protocol) IsGRPC() bool {
	switch p {
	case GRPC, GRPCS:
		return true
	default:
		return false
	}
}

func (i Protocol) IsHTTPS() bool {
	switch i {
	case HTTPS, GRPCS:
		return true
	default:
		return false
	}
}

func (p Protocol) IsDubbo() bool {
	return p == Dubbo
}

func (p Protocol) IsUnsupported() bool {
	return p == Unsupported
}

func (p Protocol) String() string {
	return string(p)
}
